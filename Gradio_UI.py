# Gradio_UI_test.py
import gradio as gr
import asyncio
import time, json, traceback
from typing import Dict, List, Tuple
from langgraph.graph import StateGraph

# 延迟导入AIMessage，避免循环导入
from langchain_core.messages.ai import AIMessage

# 声明JD_QueryState类型，避免直接从app导入
try:
    from app import JD_QueryState
except ImportError:
    from typing_extensions import TypedDict
    from typing import Annotated
    from langgraph.graph.message import add_messages

    class JD_QueryState(TypedDict):
        """State representing the user conversation of the JD product query."""
        messages: Annotated[list, add_messages]
        query: list[str]
        finished: bool

# ================== 增强消息处理器 ==================
class MessageProcessor:
    @staticmethod
    def parse(step_data: Dict) -> List[Tuple[str, str]]:
        """深度消息解析器（支持所有已知结构）"""
        messages = []
        
        # 层级1：直接消息字段
        messages.extend(MessageProcessor._parse_level(step_data.get("messages", []))) 
        
        # 层级2：chatbot节点嵌套
        chatbot_data = step_data.get("chatbot", {})
        messages.extend(MessageProcessor._parse_level(chatbot_data.get("messages", [])))  
        messages.extend(MessageProcessor._parse_level(chatbot_data.get("output", [])))  
        
        # 层级3：其他可能字段
        for key in ["output", "response", "result"]:
            messages.extend(MessageProcessor._parse_level(step_data.get(key, [])))  
        
        # 输出接收到的消息数
        msg_count = len(messages)
        assistant_count = sum(1 for role, _ in messages if role == "assistant")
        user_count = sum(1 for role, _ in messages if role == "user")
        print(f"📤 解析到{msg_count}条消息(用户:{user_count}, 助手:{assistant_count})")
        
        return messages
        
    @staticmethod
    def get_latest_assistant_message(messages: List[Tuple[str, str]]) -> str:
        """从解析的消息列表中获取最新的助手消息"""
        # 筛选出所有助手消息
        assistant_messages = [content for role, content in messages if role == "assistant"]
        
        # 返回最后一条（如果存在）
        return assistant_messages[-1] if assistant_messages else ""

    @staticmethod
    def _parse_level(msg_list: List) -> List[Tuple[str, str]]:
        """统一解析层级数据"""
        parsed = []
        for idx, item in enumerate(msg_list):
            role, content = "", ""
            log_prefix = ""
            
            # 根据消息类型分别处理
            message_type = type(item).__name__
            
            # AIMessage处理
            if isinstance(item, AIMessage):
                role, content = "assistant", item.content
                log_prefix = f"🤖 [L1] AIMessage-{idx}"
            
            # 字典消息处理
            elif isinstance(item, dict):
                role = item.get("role", "unknown").lower()
                content = str(item.get("content", ""))
                
                # 特殊处理工具消息
                if role == "tool":
                    role = "assistant"
                    content = f"工具执行结果: {content[:120]}..." if len(content) > 120 else content
                    log_prefix = f"⚙️ [L1] 工具消息-{idx}"
                elif role in ("user", "assistant"):
                    log_prefix = f"📨 [L1] 字典消息-{idx}"
            
            # 元组消息处理
            elif isinstance(item, tuple) and len(item) == 2:
                role, content = str(item[0]).lower(), str(item[1])
                log_prefix = f"📦 [L1] 元组消息-{idx}"
            
            # 添加有效的消息到结果中
            if role in ("user", "assistant") and content:
                parsed.append((role, content))
                print(f"{log_prefix}: {role}->{content[:50]}...")
        
        return parsed

# ================== 智能流式处理器 ==================
class StreamProcessor:
    def __init__(self, state_graph: StateGraph):
        self.state_graph = state_graph
        self.context = {
            "current_turn": 0,
            "history_hash": "",
            "response_cache": set(),
            "last_state": None
        }

    # +++ 新增方法 +++
    def reset_context(self):
        """重置对话上下文"""
        self.context = {
            "current_turn": 0,
            "history_hash": "",
            "response_cache": set(),
            "last_state": None
        }
        print("🔄 已重置处理器上下文")

    async def process(self, initial_state: JD_QueryState):
        """简化的对话处理器，添加了工具调用支持和超时控制"""
        try:
            # 输出代理状态
            import os
            print(f"🔐 代理状态: HTTP={'http_proxy' in os.environ}, HTTPS={'https_proxy' in os.environ}")
            for proxy_type in ['http_proxy', 'https_proxy']:
                proxy_value = os.environ.get(proxy_type, '')
                proxy_value and print(f"  {proxy_type.upper()}代理值: {proxy_value}")
            
            # 调试输出消息历史长度
            messages = initial_state.get("messages", [])
            print(f"🔍 处理状态中包含{len(messages)}条消息历史")
            
            # 打印历史消息用于调试
            if len(messages) > 1:
                print(f"📜 历史上下文概要:")
                for i, msg in enumerate(messages):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")[:30]
                    print(f"  [{i+1}] {role}: {content}...")
            
            # 检查是否包含搜索关键词
            last_query = initial_state.get("query", [""])[0]
            is_search_query = any(kw in last_query for kw in ["搜索", "查找", "寻找", "京东", "购买", "商品"])
            is_search_query and print(f"🔍 检测到搜索查询: {last_query}")
            
            # 生成一个唯一标识符避免重复处理
            fingerprint = hash(str(time.time()) + str(initial_state.get("query", [])))
            
            # 重复处理检查
            if fingerprint == self.context.get("history_hash"):
                print("⚠️ 检测到重复请求，忽略")
                return
                
            # 更新上下文
            self.context["current_turn"] = self.context.get("current_turn", 0) + 1
            self.context["history_hash"] = fingerprint
            self.context["response_cache"] = set()  # 重置响应缓存
            
            turn = self.context["current_turn"]
            print(f"🔄 开始处理第{turn}轮对话...")
            
            # 用于存储当前轮次的所有消息片段
            current_response_parts = []
            tool_called, tool_call_start_time = False, None
            step_count = 0
            
            print("🚀 准备调用state_graph.astream处理流程...")
            
            # 直接使用流式调用获取回复
            try:
                async for step in self.state_graph.astream(initial_state, {"recursion_limit": 100}):
                    step_count += 1
                    print(f"\n📍 执行步骤 {step_count} ...")
                    
                    # 调试步骤内容
                    self._debug_step_data(step)
                    
                    # 处理工具调用情况
                    if "tool_calls" in step:
                        tool_called = True
                        tool_call_start_time = time.time()
                        tool_info = step.get("tool_calls", [{}])[0]
                        tool_name = tool_info.get("name", "未知工具")
                        print(f"🔧 检测到工具调用: {tool_name}")
                        yield f"正在执行操作: {tool_name}..."
                        continue
                    
                    # 工具调用超时检查
                    if tool_called and tool_call_start_time:
                        elapsed = time.time() - tool_call_start_time
                        if elapsed > 60:  # 工具调用60秒超时
                            print(f"⚠️ 工具调用超时! 已经耗时 {elapsed:.1f} 秒")
                            yield "操作超时，可能是网络问题或京东接口暂时不可用，请稍后再试。"
                            break
                    
                    # 获取消息内容
                    step_messages = MessageProcessor.parse(step)
                    step_messages and print(f"  解析得到 {len(step_messages)} 条消息")
                    
                    # 从当前步骤中提取最新的助手消息
                    latest_assistant_msgs = self._extract_latest_assistant_message(step)
                    
                    # 处理新助手消息
                    if latest_assistant_msgs:
                        print(f"  ✓ 找到 {len(latest_assistant_msgs)} 条新助手消息")
                        for msg in latest_assistant_msgs:
                            if msg and len(msg.strip()) >= 2:  # 跳过太短的消息
                                current_response_parts.append(msg.strip())
                                print(f"  📤 输出消息: {msg.strip()[:50]}...")
                                yield msg.strip()
                    else:
                        print("  ❌ 未找到新的助手消息")
                    
                    # 检查是否需要结束
                    if step.get("finished", False):
                        print("✓ 对话已完成")
                        break
                    
                print(f"✅ 流处理完成，共执行了 {step_count} 个步骤")
                
            except Exception as stream_error:
                print(f"❌ 流处理异常: {str(stream_error)}")
                print(f"异常类型: {type(stream_error).__name__}")
                import traceback
                traceback.print_exc()
                yield f"处理过程中出现错误: {str(stream_error)[:100]}"
                    
            # 处理响应汇总输出
            if current_response_parts:
                combined = " ".join(current_response_parts)
                print(f"✓ 本轮回复汇总: {combined[:100]}...")
            else:
                print("⚠️ 未收集到任何响应片段")
                yield "抱歉，我未能生成回复。请检查网络连接或稍后重试。"
                    
        except Exception as e:
            print(f"❌ 处理器异常: {str(e)}")
            traceback.print_exc()  # 打印详细错误信息
            yield f"很抱歉，处理您的请求时出现问题。({str(e)[:50]})"
            
        finally:
            print(f"✓ 第{self.context.get('current_turn', 0)}轮对话处理完成")
            
    def _debug_step_data(self, step):
        """调试步骤数据的辅助方法"""
        try:
            step_keys = list(step.keys()) if isinstance(step, dict) else "非字典类型"
            print(f"  步骤数据类型: {type(step).__name__}, 包含键: {step_keys}")
            
            # 处理消息数据
            if isinstance(step, dict) and "messages" in step:
                msgs = step["messages"]
                print(f"  消息列表长度: {len(msgs)}")
                if msgs:
                    last_msg = msgs[-1]
                    msg_role = last_msg.get("role", "未知") if isinstance(last_msg, dict) else type(last_msg).__name__
                    msg_content = str(last_msg.get("content", ""))[:50] if isinstance(last_msg, dict) else str(last_msg)[:50]
                    print(f"  最新消息: {msg_role} -> {msg_content}...")
            
            # 检查节点信息
            isinstance(step, dict) and "node_name" in step and print(f"  当前节点: {step['node_name']}")
            
            # 检查完成状态
            isinstance(step, dict) and "finished" in step and print(f"  完成状态: {step['finished']}")
            
        except Exception as detail_e:
            print(f"  ⚠️ 解析步骤详情失败: {str(detail_e)}")

    def _extract_latest_assistant_message(self, step_data: Dict) -> List[str]:
        """从步骤数据中提取最新的助手消息，优化工具调用处理"""
        # 处理工具调用
        if "tool_calls" in step_data:
            tool_calls = step_data.get("tool_calls", [])
            if not tool_calls:
                return []
                
            # 提取工具信息并返回消息
            try:
                tool_info = tool_calls[0] if isinstance(tool_calls, list) else tool_calls
                tool_name = tool_info.get("name", "未知工具")
                tool_args = json.dumps(tool_info.get("args", {}), ensure_ascii=False)
                
                # 构建工具调用消息
                tool_msg = f"正在使用{tool_name}工具，参数: {tool_args[:50]}..."
                
                msg_hash = hash(tool_msg)
                if msg_hash not in self.context.get("response_cache", set()):
                    self.context.setdefault("response_cache", set()).add(msg_hash)
                    return [tool_msg]
            except Exception as e:
                print(f"⚠️ 解析工具调用出错: {str(e)}")
            
            return []
            
        # 处理工具响应    
        if "tool_response" in step_data:
            try:
                response = step_data.get("tool_response", {})
                content = response.get("content", "")
                
                if content:
                    msg_hash = hash(content)
                    if msg_hash not in self.context.get("response_cache", set()):
                        self.context.setdefault("response_cache", set()).add(msg_hash)
                        return [f"工具执行结果: {str(content)[:200]}..."]
            except Exception as e:
                print(f"⚠️ 解析工具响应出错: {str(e)}")
            
            return []
        
        # 检查完成状态
        step_data.get("finished", False) and print("✓ 检测到对话完成标志")
        
        # 处理普通消息
        parsed_messages = MessageProcessor.parse(step_data)
        
        if not parsed_messages:
            return []
            
        # 获取最新的助手消息
        latest_message = MessageProcessor.get_latest_assistant_message(parsed_messages)
        if not latest_message:
            return []
            
        # 检查是否已处理过
        msg_hash = hash(latest_message)
        if msg_hash in self.context.get("response_cache", set()):
            return []
            
        # 添加到缓存并返回
        self.context.setdefault("response_cache", set()).add(msg_hash)
        return [latest_message]

    async def _process_with_timeout(self, task, timeout_seconds):
        """带超时控制的异步迭代器包装器"""
        accumulated_chunks = []
        start_time = time.time()
        print(f"⏱️ 启动超时保护，最大处理时间: {timeout_seconds}秒")
        
        try:
            # 处理任务流
            chunk_count = 0
            
            async for chunk in task:
                chunk_count += 1
                current_time = time.time()
                elapsed = current_time - start_time
                
                # 记录第一个响应时间
                chunk_count == 1 and print(f"⏱️ 收到第一个响应，耗时: {elapsed:.2f}秒")
                
                # 超时检查
                if elapsed > timeout_seconds:
                    print(f"⚠️ 处理超时! 已经耗时 {elapsed:.1f} 秒")
                    raise asyncio.TimeoutError(f"处理时间超过{timeout_seconds}秒")
                
                # 收集并返回块
                accumulated_chunks.append(chunk)
                yield chunk
                
            # 记录完成情况
            completion_time = time.time() - start_time
            print(f"✅ 处理成功完成，总耗时: {completion_time:.2f}秒，处理了{len(accumulated_chunks)}个响应片段")
                
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"⚠️ 处理超时! 已经耗时 {elapsed:.1f} 秒，已收集 {len(accumulated_chunks)} 个回复片段")
            
            # 返回部分结果
            accumulated_chunks and (yield "由于操作时间过长，处理被中断。以下是已获取的部分信息：")
            raise
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ 处理过程中出现异常: {str(e)}")
            print(f"  处理已进行 {elapsed:.1f} 秒，收集了 {len(accumulated_chunks)} 个片段")
            import traceback
            traceback.print_exc()
            
            # 返回部分结果
            accumulated_chunks and (yield "处理过程中出现错误，以下是部分结果：")
            raise

# ================== 稳健界面系统 ==================
class GradioUI:
    def __init__(self, state_graph: StateGraph):
        self.state_graph = state_graph
        self.processor = StreamProcessor(state_graph)
        self.interface = self._build_interface()
        
    # 添加代理管理相关方法
    def _enable_proxy(self):
        """启用HTTP和HTTPS代理"""
        import os
        # 为LLM调用设置代理
        os.environ["http_proxy"] = "http://127.0.0.1:7890"
        os.environ["https_proxy"] = "http://127.0.0.1:7890"
        print("✅ 已启用网络代理")

    def _disable_proxy(self):
        """禁用HTTP和HTTPS代理"""
        import os
        # 暂时清除代理设置
        if "http_proxy" in os.environ:
            del os.environ["http_proxy"]
        if "https_proxy" in os.environ:
            del os.environ["https_proxy"]
        print("❌ 已禁用网络代理，本地服务可正常启动")

    def _build_interface(self):
        """构建抗卡顿界面"""
        import gradio as gr
        
        # 设置环境变量代替直接配置
        import os
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
        
        with gr.Blocks(title="京东导购助手", theme=gr.themes.Soft(
            primary_hue="emerald",
            secondary_hue="gray"
        ), css="""
        body {
            background-color: #f7f7f7;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 15px;
            box-sizing: border-box;
        }
        .gradio-container {
            width: 85vw !important;
            max-width: 1400px !important;
            min-width: 800px !important;
            margin: 0 auto !important;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .main-div {
            display: flex;
            flex-direction: column;
            width: 100%;
            padding: 15px;
        }
        .loading {color: #888 !important; font-style: italic}
        .message-wrap {width: 95% !important;}
        .message-wrap .bubble {max-width: 85% !important;}
        .chatbot {
            height: 60vh !important; 
            min-height: 400px !important;
            overflow-y: auto;
        }
        .input-row {border-radius: 8px; overflow: hidden; margin-top: 8px;}
        .button-column {display: flex; flex-direction: column; gap: 8px;}
        
        /* 头部布局 */
        .header-area {margin-bottom: 10px !important;}
        h1 {margin: 0 !important; padding: 10px 0 5px 0 !important; font-size: 24px !important;}
        p {margin: 0 0 5px 0 !important;}
        
        /* 底部区域 */
        .footer-area {
            margin-top: 10px;
            border-top: 1px solid #eee;
            padding-top: 10px;
        }
        
        /* 强制使用指南始终显示 */
        .guide-section {
            margin: 10px 0 !important;
            border: 1px solid #eaeaea;
            border-radius: 8px;
            background-color: #f8f9fa;
            padding: 8px;
        }
        .guide-section > .hide {display: block !important;}
        .guide-section > .label-wrap {cursor: default !important;}
        .guide-section > .label-wrap > svg {display: none !important;}
        
        /* 底部状态区域 */
        .footer {
            text-align: center;
            margin-top: 5px;
            font-size: 0.8em;
            color: #666;
        }
        
        /* 响应式调整 */
        @media screen and (max-height: 700px) {
            .chatbot {height: 50vh !important;}
            h1 {font-size: 20px !important;}
        }
        @media screen and (max-width: 1000px) {
            .gradio-container {width: 95vw !important;}
        }
        """, analytics_enabled=False) as interface:
            # 页头 - 添加类名便于样式控制
            with gr.Column(elem_classes="header-area"):
                gr.Markdown("""<div style="text-align: center;">
                    <h1>🛍️ 京东智能导购</h1>
                    <p>您好！我是您的私人购物助手，随时为您推荐优质商品</p>
                </div>""")
            
            # 聊天区
            with gr.Column(scale=1, min_width=800):
                chatbot = gr.Chatbot(
                    label="",
                    avatar_images=(
                        "https://cdn-icons-png.flaticon.com/512/1077/1077114.png",  # 用户头像
                        "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"    # 助手头像
                    ),
                    show_copy_button=True,
                    type="messages",  # 设置消息类型为messages
                    render_markdown=True,  # 启用Markdown渲染
                    sanitize_html=False,   # 允许HTML内容 
                    height=460,            # 调整高度以适应固定容器
                    elem_classes="chatbot"
                )

                # 输入区
                with gr.Row(variant="panel", elem_classes="input-row"):
                    input_box = gr.Textbox(
                        placeholder="请输入问题，例如：推荐手机...",
                        container=False,
                        scale=7,
                        autofocus=True,
                        max_lines=3,
                        elem_classes="input-box"
                    )
                    with gr.Column(scale=1, elem_classes="button-column"):
                        submit_btn = gr.Button("🚀 发送", variant="primary", size="sm")
                        clear_btn = gr.Button("🧹 清空", variant="secondary", size="sm")
            
            # 底部区域
            with gr.Column(elem_classes="footer-area"):
                # 添加使用指南到底部
                with gr.Accordion("使用指南", open=True, elem_classes="guide-section"):
                    gr.Markdown("""
                    ### 🛍️ 京东导购助手使用指南
                    
                    1. **搜索商品**: 输入"搜索 [商品名称]"即可查找商品
                    2. **比较商品**: 可以询问不同商品的区别和优缺点
                    3. **获取建议**: 描述您的需求，AI会给出适合的推荐
                    4. **查看详情**: 点击商品链接可以直接跳转到京东购买
                    
                    > 提示：搜索结果会以表格形式展示，包含产品图片和购买链接
                    """)
                
                # 添加状态信息
                with gr.Row(elem_classes="footer"):
                    gr.Markdown("""<div style="text-align: center; margin-top: 5px; font-size: 0.8em; color: #666; width: 100%;">
                        系统状态: 已连接 | 点击发送按钮开始对话
                    </div>""")
            
            # 交互逻辑
            submit_btn.click(
                self._user_input_handler,
                [input_box, chatbot],
                [input_box, chatbot]
            ).then(
                self._bot_response_handler,
                chatbot,
                chatbot
            )
            
            input_box.submit(
                self._user_input_handler,
                [input_box, chatbot],
                [input_box, chatbot]
            ).then(
                self._bot_response_handler,
                chatbot,
                chatbot
            )
            
            clear_btn.click(
                lambda: [],  # 返回空列表清除历史
                None, 
                chatbot,
                queue=False
            )
                
        # 设置队列（简化版本）
        self._setup_queue(interface)
                
        return interface
        
    def _setup_queue(self, interface):
        """设置队列兼容不同版本"""
        queue_config = {"concurrency_count": 5, "max_size": 20}
        
        try:
            # 直接尝试设置队列
            interface.queue(**queue_config)
            return True
        except Exception as e:
            print(f"无法使用新版队列API: {str(e)}")
            
            # 尝试兼容模式
            try:
                interface.queue()  # 不带参数的调用
                print("使用兼容模式设置队列")
                return True
            except:
                print("警告: 无法设置队列，UI响应可能较慢")
                return False

    def _user_input_handler(self, user_input: str, history: list):
        """用户输入处理（带消毒）"""
        # 当开启新对话时重置处理器
        len(history) == 0 and self.processor.reset_context()
    
        sanitized_input = user_input.strip()[:500]
        print(f"\n👤 用户输入（{len(sanitized_input)}字）：{sanitized_input[:50]}...")
        
        # 构建兼容 messages 类型的消息格式
        new_message = {"role": "user", "content": sanitized_input}
        
        # 返回空文本框和更新的历史记录
        return "", history + [new_message]

    async def _bot_response_handler(self, history: list):
        """简化的响应处理器，兼容messages类型"""
        if not history:
            print("❌ 错误: 空历史记录")
            yield []
            return
            
        try:
            # 确保启用代理以便LLM调用
            self._enable_proxy()
            print("✅ 已确保代理启用")
            
            # 获取最后一条用户消息
            last_message = history[-1]
            if last_message["role"] != "user":
                print("❌ 错误: 最后一条消息不是用户消息")
                yield history
                return
                
            last_user_message = last_message["content"]
            print(f"用户问题: {last_user_message}")
            
            # 检查是否触发了搜索关键词
            search_keywords = ["搜索", "查找", "找一下", "寻找", "查询", "找找"]
            is_search_query = any(keyword in last_user_message for keyword in search_keywords)
            
            # 构建完整的消息历史
            full_messages = []
            for msg in history[:-1]:  # 不包括最后一条待处理的消息
                # 转换为app.py中JD_QueryState接受的格式
                role = msg.get("role", "user")
                content = msg.get("content", "")
                full_messages.append({"role": role, "content": content})
                
            # 添加最后一条用户消息
            full_messages.append({"role": "user", "content": last_user_message})
            
            print(f"📚 构建了完整历史上下文，共{len(full_messages)}条消息")
            
            # 创建带有完整历史的状态
            initial_state = JD_QueryState(
                messages=full_messages,
                query=[last_user_message],  # 当前查询仍然只包含最新消息
                finished=False
            )
            
            print(f"🔄 创建初始状态: {str(initial_state)[:100]}...")
            
            # 标记当前响应为处理中
            thinking_content = "正在搜索商品，请稍候...这可能需要一点时间" if is_search_query else "思考中..."
            history.append({"role": "assistant", "content": thinking_content})
            yield history
            
            print("⌛ 显示思考中状态，开始处理...")
            
            # 收集完整响应
            full_response = ""
            response_success = False
            
            # 带超时的流式处理
            timeout = 120  # 统一超时时间
            print(f"📡 准备调用处理器...设置超时时间: {timeout}秒")
            
            try:
                process_task = self.processor.process(initial_state)
                response_chunk_count = 0
                
                async for chunk in self.processor._process_with_timeout(process_task, timeout):
                    response_chunk_count += 1
                    
                    # 处理有效片段
                    chunk_text = chunk.strip()
                    if chunk_text:
                        # 添加到完整响应
                        full_response += chunk_text + " "
                        # 更新历史并显示
                        history[-1] = {"role": "assistant", "content": full_response.strip()}
                        print(f"📤 接收到第{response_chunk_count}个响应片段: {chunk_text[:30]}...")
                        yield history
                        # 轻微延迟使界面更流畅
                        await asyncio.sleep(0.05)
                
                print(f"✅ 流处理完成，共接收{response_chunk_count}个响应片段")
                response_success = True
                
            except (asyncio.TimeoutError, Exception) as process_error:
                # 错误类型标识
                error_type = "超时" if isinstance(process_error, asyncio.TimeoutError) else "异常"
                print(f"❌ 处理{error_type}: {str(process_error)}")
                
                # 记录详细错误信息
                isinstance(process_error, asyncio.TimeoutError) or traceback.print_exc()
                
                # 构建错误消息
                error_prefix = "搜索操作超时。京东搜索可能暂时不可用，请稍后再试。" if isinstance(process_error, asyncio.TimeoutError) else f"处理请求时出错: {str(process_error)[:100]}"
                
                # 构建最终响应
                final_content = full_response.strip() + "\n\n" + error_prefix if full_response else error_prefix
                history[-1] = {"role": "assistant", "content": final_content}
                yield history
                return
                
            # 最终响应处理
            if response_success:
                clean_response = full_response.strip()
                
                # 根据响应情况设置不同的消息
                if not clean_response:
                    fallback_msg = "抱歉，我无法完成商品搜索。可能是网络问题或者京东接口暂时不可用。请稍后再试。" if is_search_query else "抱歉，我没能找到相关信息，请换个问题试试。"
                    history[-1] = {"role": "assistant", "content": fallback_msg}
                    print("⚠️ 未收集到任何有效响应")
                else:
                    history[-1] = {"role": "assistant", "content": clean_response}
                    print(f"✅ 成功生成回复，长度:{len(clean_response)}")
                
                print(f"回复: {history[-1]['content'][:50]}...")
                yield history
                
        except Exception as e:
            # 处理最外层异常
            print(f"❌ 处理器整体异常: {str(e)}")
            traceback.print_exc()
            
            # 尝试添加错误响应
            error_msg = f"处理您的请求时出错，请重试。错误: {str(e)[:100]}"
            history and history.append({"role": "assistant", "content": error_msg})
            
            yield history

    def _init_environment(self):
        """初始化运行环境，处理代理冲突"""
        import os
        
        # 设置NO_PROXY环境变量以避免代理冲突
        local_addresses = ["localhost", "127.0.0.1", "0.0.0.0", "::1"]
        os.environ["NO_PROXY"] = ",".join(local_addresses)
        os.environ["no_proxy"] = os.environ["NO_PROXY"]
        print(f"🔒 已设置NO_PROXY={os.environ['NO_PROXY']}，确保本地连接不经过代理")
        
        # 删除可能干扰的代理设置
        for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY"]:
            proxy_var in os.environ and os.environ.pop(proxy_var)
        
        # 设置Gradio环境变量
        gradio_env_settings = {
            "GRADIO_ANALYTICS_ENABLED": "False",  # 禁用分析
            "GRADIO_ALLOW_FLAGGING": "never",     # 禁用标记
            "GRADIO_THEME": "soft",               # 使用软主题
            "GRADIO_USE_PROXY": "false"           # 禁用Gradio代理
        }
        
        # 批量应用环境变量
        for key, value in gradio_env_settings.items():
            os.environ[key] = value
        
        # 返回代理状态
        return "http_proxy" in os.environ or "https_proxy" in os.environ

    def launch(self, **kwargs):
        """启动稳健服务"""
        # 初始化环境
        self._init_environment()
        
        # 设置启动参数，合并默认参数和传入参数
        launch_kwargs = {
            # 默认参数
            "server_name": "0.0.0.0",  # 使用0.0.0.0允许从任何IP访问
            "server_port": 7861,       # 使用固定端口
            "share": False,            # 默认不分享
            "inbrowser": True,         # 自动打开浏览器
            "max_threads": 30,         # 更多线程
            "quiet": False,            # 显示所有日志
            **kwargs                   # 合并传入的参数
        }
        
        print(f"🚀 正在启动UI服务，参数: {launch_kwargs}")
        
        # 临时禁用代理以启动服务器
        self._disable_proxy()
        
        # 确保启动后恢复代理
        result = None
        try:
            result = self.interface.launch(**launch_kwargs)
        except Exception as e:
            print(f"❌ 启动出错: {str(e)}")
            raise
        finally:
            # 无论成功与否，都恢复代理设置
            self._enable_proxy()
            
        return result

    @staticmethod
    def is_search_query(query: str) -> bool:
        """检查是否是搜索查询"""
        search_keywords = ["搜索", "查找", "找一下", "寻找", "查询", "找找", "京东", "购买", "商品"]
        product_keywords = ["手机", "电脑", "笔记本", "ROG", "华为", "苹果", "小米", "电视", "耳机"]
        
        # 检查是否包含搜索关键词
        has_search_kw = any(kw in query for kw in search_keywords)
        
        # 检查是否包含产品关键词
        has_product_kw = any(kw in query for kw in product_keywords)
        
        # 同时包含搜索关键词和产品关键词的可能是搜索请求
        return has_search_kw and has_product_kw

if __name__ == "__main__":
    # 导入必要模块
    import os, sys, traceback
    
    try:
        # 创建UI实例
        from app import graph_with_tools
        ui = GradioUI(graph_with_tools)
        
        # 配置环境
        # 设置NO_PROXY环境变量
        local_addresses = ["localhost", "127.0.0.1", "0.0.0.0", "::1"]
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(local_addresses)
        print(f"🔒 已设置NO_PROXY={os.environ['NO_PROXY']}，确保本地连接不经过代理")
        
        # 环境准备
        ui._disable_proxy()
        print("✅ 临时取消代理设置，防止本地访问冲突")
        
        # 设置Gradio环境变量
        os.environ.update({
            "GRADIO_ANALYTICS_ENABLED": "False",
            "GRADIO_USE_PROXY": "false"
        })
        
        # 显示版本信息
        print(f"✓ Gradio版本: {gr.__version__}")
        print("✅ 成功导入图对象")
        
        # 启动UI
        ui.launch()
        
    except Exception as e:
        print(f"\n❌ 启动失败: {str(e)}")
        print("\n--- 详细错误信息 ---")
        traceback.print_exc()
        print("""
        ===== 解决建议 =====
        1. 检查是否安装了所有依赖: pip install -r requirements.txt
        2. 验证gradio安装: pip install --upgrade gradio
        3. 检查防火墙是否允许本地端口访问
        4. 尝试其他端口: export GRADIO_SERVER_PORT=8000
        5. 如果使用Clash，尝试在Clash设置中添加"localhost,127.0.0.1"到绕过代理列表
        """)
    finally:
        # 确保恢复代理设置
        try:
            ui._enable_proxy()
            print("✓ 代理设置已恢复")
        except:
            pass
