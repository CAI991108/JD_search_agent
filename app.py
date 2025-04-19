from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class JD_QueryState(TypedDict):
    """State representing the user conversation of the JD product query."""
    messages: Annotated[list, add_messages]
    query: list[str]
    finished: bool


JD_QueryBot_SYSINT = (
    "system",
    "你是京东导购小助手，一个熟悉京东商场和各种产品的智能系统。"
    "你非常善用京东的搜索引擎，能够精准定位到用户所需的产品。"
    "用户会向你咨询各种产品，你会根据用户的需求，推荐合适的产品。"
    "你可以结合用户咨询的问题和京东的搜索引擎，给出推荐的产品及理由分析。"
    "你善于将用户的需求转化为产品推荐，并做成表格的形式展示给用户。"
    "\n\n"
    "你的基础大语言模型是调用client=OpenAI的qwen/qwen2.5-vl-32b-instruct:free"
    "你拥有非常好的基础语料库，擅长思考并能够理解用户的需求，同时具备一定的视觉能力。"
    "使用JD_search_general(search_keyword)函数可以打开浏览器使用京东的搜索。"
    "search_keyword需要你从用户的提问中提取归纳出来，一般客户的提问中会包含产品的名称、品牌、价格区间等信息。"
    "JD_search_general()函数最终会保存成一个jd_product_details.json文件，包含商品标题、价格、图片链接、购买链接等信息。"
    "生成的json文件会被保存在当前目录下，你可以通过read_product_details(file_path)函数读取json文件，"
    "部分产品的具体信息可能会在图片中，你将运用你的视觉能力，提取出产品有关的具体信息。"
    "你可以用工具函数 extract_text_from_image_url(image_url)来帮助你完成图片内容提取的任务。"
    "注意：京东上面的部分产品可能有国家补贴，实际价格以国补后为准。"
    "保存的商品信息内容可能不会太完整，你可以结合自身的语料库知识，或者使用其他搜索引擎收集相关内容来补充。"
    "\n\n"
    "如果任何工具不可用，或者客户的问题不明确，任何报错你都可以打破次元壁随时反馈给用户。"
)

WELCOME_MSG = (
    "欢迎使用京东导购小助手！"
    "请问有什么我可以帮助你的吗？"
    "你可以问我任何关于京东产品的问题，我会尽力为你推荐合适的产品。"
    "如果你需要了解更多产品信息，请告诉我，我会帮你搜索并提供详细信息。"
)


from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from llama_index.core.base.llms.types import ChatMessage
from langchain_core.messages.ai import AIMessage
from typing import Literal
from langchain_core.tools import tool
from playwright.async_api import async_playwright
import asyncio, random, json, nest_asyncio, time
from openai import OpenAI

########上传Kaggle前注释掉########
import os
# 设置 HTTP 和 HTTPS 代理
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# 设置NO_PROXY环境变量以避免代理冲突
local_addresses = ["localhost", "127.0.0.1", "0.0.0.0", "::1"]
os.environ["NO_PROXY"] = ",".join(local_addresses)
os.environ["no_proxy"] = os.environ["NO_PROXY"]

secret_value_0 = <your_GOOGLE_API_KEY_here>

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=secret_value_0)

# test_app.py 关键修改部分

# 修改chatbot函数的消息处理
def chatbot(state: JD_QueryState) -> JD_QueryState:
    """The chatbot itself. A simple wrapper around the model's own chat interface."""
    message_history = [JD_QueryBot_SYSINT] + state["messages"]
    
    # 确保返回的消息格式统一
    response = llm.invoke(message_history)
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": response.content}],
        "query": state["query"],
        "finished": state["finished"]
    }

# 修改human_node的消息格式
def human_node(state: JD_QueryState) -> JD_QueryState:
    """直接从状态中获取用户输入（无需终端交互）"""
    # 确保消息列表中最后一条是用户输入
    user_input = state["query"][-1] if state["query"] else ""
    
    # 更新状态（直接传递用户输入）
    return {
        "messages": state["messages"] + [{"role": "user", "content": user_input}],
        "query": state["query"],
        "finished": state["finished"]
    }

# 修改chatbot_with_welcome_msg的消息处理
def chatbot_with_welcome_msg(state: JD_QueryState) -> JD_QueryState:
    """The chatbot with welcome message."""
    if state["messages"]:
        new_msg = llm.invoke([JD_QueryBot_SYSINT] + state["messages"])
    else:
        new_msg = AIMessage(content=WELCOME_MSG)
    
    # 统一转换为字典格式
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": new_msg.content}],
        "query": state["query"],
        "finished": state["finished"]
    }

def chatbot_with_tools(state: JD_QueryState) -> JD_QueryState:
    """确保每次响应后终止对话"""
    print(f"🤖 Chatbot节点: 消息数量={len(state.get('messages', []))}")
    
    # 检测到对话已完成则直接返回
    if state.get("finished", False):
        print("🛑 检测到对话已完成标记，不进行LLM调用")
        return state
    
    # 默认响应值
    response_content = WELCOME_MSG
    response_dict = {"role": "assistant", "content": response_content}
    is_finished = True  # 默认标记为完成
    
    # 检查代理设置
    import os
    print(f"🔐 LLM调用前代理状态: HTTP={'http_proxy' in os.environ}, HTTPS={'https_proxy' in os.environ}")
    
    # 处理有消息的情况
    if state["messages"]:
        try:
            print("🚀 调用LLM处理用户消息...")
            new_output = llm_with_tools.invoke([JD_QueryBot_SYSINT] + state["messages"])
            
            # 提取响应内容
            response_content = getattr(new_output, "content", str(new_output))
            print(f"✅ LLM响应成功: {response_content[:50]}...")
            
            # 检测工具调用
            has_tool_calls = hasattr(new_output, "tool_calls") and new_output.tool_calls
            
            # 根据响应类型返回不同结果
            if has_tool_calls:
                print(f"🔧 检测到LLM响应中包含工具调用 (AIMessage格式)")
                return {
                    "messages": state["messages"] + [new_output],
                    "query": state.get("query", []),
                    "finished": False  # 允许工具处理
                }
            
            # 普通响应处理
            response_dict = {"role": "assistant", "content": response_content}
            
        except Exception as e:
            print(f"❌ LLM调用异常: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 设置错误消息
            response_dict = {
                "role": "assistant", 
                "content": f"对不起，处理您的请求时出现问题: {str(e)[:100]}"
            }
    else:
        print("ℹ️ 使用欢迎消息响应")
    
    # 返回更新的状态
    return {
        "messages": state["messages"] + [response_dict],
        "query": state.get("query", []),
        "finished": is_finished  # 标记对话完成
    }


@tool
def JD_search_general(search_keyword: str) -> str:
    """
    使用 Playwright 打开浏览器并在京东官网搜索指定关键词。

    Args:
        search_keyword (str): 要搜索的关键词。

    Returns:
        str: 包含搜索结果的商品信息并保存为 JSON 文件。
    """
    import os
    import json
    import sys
    
    print("=" * 50)
    print(f"🚀 开始执行京东搜索工具: 关键词='{search_keyword}'")
    print(f"🔍 Python版本: {sys.version}")
    
    # 检查playwright依赖
    try:
        import playwright
    except ImportError:
        error_message = "错误: 需要安装Playwright。请运行: pip install playwright && playwright install chromium"
        print(f"❌ {error_message}")
        return json.dumps([{"title": error_message, "price": "N/A"}], ensure_ascii=False)
    
    # 代理处理
    proxies = handle_proxy_settings()
    
    try:
        # 确保cookie文件存在
        cookies_file = ensure_cookies_file()
        
        # 执行搜索流程
        search_results = execute_jd_search(search_keyword, cookies_file)
        
        # 返回JSON格式的结果
        return json.dumps(search_results, ensure_ascii=False, indent=4)
    finally:
        # 恢复代理设置
        restore_proxy_settings(proxies)

def handle_proxy_settings():
    """临时禁用代理并保存原始设置"""
    import os
    
    # 保存原始代理设置
    http_proxy = os.environ.pop("http_proxy", None)
    https_proxy = os.environ.pop("https_proxy", None)
    
    print("ℹ️ 临时取消代理设置以访问京东")
    print(f"🔐 当前代理状态: HTTP={'http_proxy' in os.environ}, HTTPS={'https_proxy' in os.environ}")
    
    return {"http": http_proxy, "https": https_proxy}

def restore_proxy_settings(proxies):
    """恢复原始代理设置"""
    import os
    
    # 恢复代理设置
    if proxies["http"]:
        os.environ["http_proxy"] = proxies["http"]
    if proxies["https"]:
        os.environ["https_proxy"] = proxies["https"]
    print("ℹ️ 已恢复原始代理设置")

def ensure_cookies_file():
    """确保京东cookie文件存在"""
    import os
    import json
    
    cookies_file = "jd_cookies.json"
    
    # 文件已存在则直接返回
    if os.path.exists(cookies_file):
        return cookies_file
        
    # 创建空cookie文件
    print(f"⚠️ 京东Cookie文件'{cookies_file}'不存在，将创建一个空的Cookie文件")
    try:
        with open(cookies_file, "w", encoding="utf-8") as f:
            json.dump([], f)
        print(f"✅ 已创建空Cookie文件: {cookies_file}")
        return cookies_file
    except Exception as e:
        error_message = f"错误: 无法创建Cookie文件: {str(e)}"
        print(error_message)
        raise Exception(error_message)

def execute_jd_search(search_keyword, cookies_file):
    """执行京东搜索流程"""
    import json
    import asyncio
    import nest_asyncio

    try:
        # 设置全局超时
        print(f"🔍 开始执行搜索京东商品: {search_keyword}")
        nest_asyncio.apply()
        
        # 执行搜索，添加超时保护
        try:
            # 修正函数名使用，确保与实际定义的异步函数名匹配
            search_func = jd_search_general
            jd_product_details = asyncio.run(asyncio.wait_for(
                search_func(search_keyword, cookies_file), 
                timeout=120.0
            ))
            print(f"✅ 搜索完成，获取到{len(jd_product_details)}个商品信息")
            return jd_product_details
        except asyncio.TimeoutError:
            error_message = f"搜索'{search_keyword}'操作超时(120秒)，请稍后再试"
            print(f"⚠️ {error_message}")
            return [{"title": error_message, "price": "N/A"}]
        
    except Exception as e:
        error_message = f"搜索'{search_keyword}'时发生错误: {str(e)}"
        print(f"❌ {error_message}")
        return [{"title": error_message, "price": "N/A"}]


@tool
def read_product_details(file_path: str) -> str:
    """
    读取JD_search_general生成的 JSON 文件并返回内容。

    Args:
        file_path (str): JSON 文件的路径。

    Returns:
        str: JSON 文件的内容。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        product_details = json.load(f)
    return product_details


@tool
def extract_text_from_image_url(image_url: str) -> str:
    """
    调用 OpenRouter API 提取图片 URL 中的文字信息
    image_url 可以是来自 JD_search_general() 函数的返回结果，
    也可以通过read_product_details()函数读取的json文件中的图片链接。
    Args:
        image_url (str): 图片的 URL 地址。
    Returns:
        str: 提取的文字信息。
    """
    # 初始化 OpenAI 客户端
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=<your_OpenRouter_API_KEY_here>)

    # 构造请求消息
    messages = [{"role": "user","content": 
                [{"type": "text", "text": "仅提取图片中的文字内容，不要添加任何其他信息。"},
                {"type": "image_url", "image_url": {"url": image_url}}]}]

    # 调用 OpenRouter API
    completion = client.chat.completions.create(model="qwen/qwen2.5-vl-32b-instruct:free", messages=messages)

    # 提取响应中的文字内容
    return completion.choices[0].message.content.strip()


from langgraph.prebuilt import ToolNode

tools = [JD_search_general, read_product_details, extract_text_from_image_url]

# 修改ToolNode的消息处理
# 修改 tool_node 函数
# 在工具节点返回更易读的内容
def tool_node(state: JD_QueryState) -> JD_QueryState:
    """Execute tools and return results."""
    # 获取最后一个消息
    last_message = state["messages"][-1]
    print(f"🔍 工具节点接收到消息: {last_message}")
    
    # 从不同类型的消息中获取工具调用
    tool_calls = []
    
    # 处理字典类型消息
    if isinstance(last_message, dict):
        tool_calls = last_message.get("tool_calls", [])
        print(f"📦 从字典中提取工具调用: {len(tool_calls)}个")
        return process_tool_calls(state, tool_calls)
    
    # 处理AIMessage或其他对象类型
    tool_calls = extract_tool_calls_from_object(last_message)
    return process_tool_calls(state, tool_calls)

def extract_tool_calls_from_object(message_obj):
    """从消息对象中提取工具调用"""
    tool_calls = []
    
    try:
        # 检查直接的tool_calls属性
        if hasattr(message_obj, "tool_calls") and message_obj.tool_calls:
            for tool_call in message_obj.tool_calls:
                tool_call_info = extract_single_tool_call(tool_call)
                if tool_call_info:
                    tool_calls.append(tool_call_info)
                    print(f"✅ 成功提取工具调用: {tool_call_info['name']}")
        
        # 检查OpenAI旧格式的function_call
        if not tool_calls and hasattr(message_obj, "additional_kwargs"):
            add_kwargs = message_obj.additional_kwargs
            if isinstance(add_kwargs, dict) and "function_call" in add_kwargs:
                func_call = add_kwargs["function_call"]
                if isinstance(func_call, dict) and "name" in func_call and "arguments" in func_call:
                    try:
                        import json
                        args = json.loads(func_call["arguments"])
                        tool_calls.append({
                            "name": func_call["name"],
                            "args": args
                        })
                        print(f"✅ 从function_call中提取工具调用: {func_call['name']}")
                    except Exception as e:
                        print(f"⚠️ 解析function_call参数时出错: {str(e)}")
    except Exception as e:
        print(f"⚠️ 提取工具调用时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        
    return tool_calls

def extract_single_tool_call(tool_call):
    """从单个工具调用对象中提取信息"""
    tool_call_dict = {}
    
    # 方式1: 标准格式 - 直接有name和args属性
    if hasattr(tool_call, "name") and hasattr(tool_call, "args"):
        tool_call_dict = {
            "name": tool_call.name,
            "args": tool_call.args
        }
    # 方式2: 有name和args作为字典项
    elif isinstance(tool_call, dict) and "name" in tool_call and "args" in tool_call:
        tool_call_dict = {
            "name": tool_call["name"],
            "args": tool_call["args"]
        }
    # 方式3: 处理特殊的Gemini/Google AI格式
    elif hasattr(tool_call, "type") and getattr(tool_call, "type") == "tool_call":
        if hasattr(tool_call, "name") and hasattr(tool_call, "args"):
            tool_call_dict = {
                "name": tool_call.name,
                "args": tool_call.args
            }
    
    return tool_call_dict if tool_call_dict and "name" in tool_call_dict else None

def process_tool_calls(state: JD_QueryState, tool_calls: list) -> JD_QueryState:
    """处理提取到的工具调用"""
    tool_outputs = []
    
    print(f"⚙️ 工具调用节点接收到 {len(tool_calls)} 个工具调用请求")
    
    # 如果没有工具调用，返回错误消息
    if not tool_calls:
        error_msg = "工具调用过程中出现问题，未能获取结果。"
        print(f"⚠️ {error_msg}")
        return {
            "messages": state["messages"] + [{"role": "assistant", "content": error_msg}],
            "query": state["query"],
            "finished": True
        }
    
    for tool_call in tool_calls:
        # 提取工具信息
        tool_info = get_tool_info(tool_call)
        tool_name, args = tool_info["name"], tool_info["args"]
        print(f"🔧 正在执行工具: {tool_name}, 参数: {str(args)[:100]}...")
        
        # 查找并执行工具
        tool_output = execute_tool(tool_name, args)
        tool_outputs.append(tool_output)
    
    # 返回结果并标记为已完成
    return {
        "messages": state["messages"] + tool_outputs,
        "query": state["query"],
        "finished": True
    }

def get_tool_info(tool_call):
    """提取工具名称和参数"""
    if isinstance(tool_call, dict):
        return {
            "name": tool_call.get("name", "未知工具"),
            "args": tool_call.get("args", {})
        }
    else:
        return {
            "name": getattr(tool_call, "name", "未知工具"),
            "args": getattr(tool_call, "args", {})
        }

def execute_tool(tool_name: str, args: dict):
    """执行工具调用并返回结果"""
    # 查找工具
    try:
        tool = next((t for t in tools if t.name == tool_name), None)
        
        # 工具不存在的情况
        if not tool:
            error_msg = f"错误: 找不到名为'{tool_name}'的工具"
            print(f"❌ {error_msg}")
            return {"role": "assistant", "content": error_msg}
        
        print(f"✅ 找到工具: {tool_name}")
        
        # 执行工具调用
        start_time = time.time()
        
        # 检查代理设置
        import os
        print(f"🔐 工具执行前代理状态: HTTP={'http_proxy' in os.environ}, HTTPS={'https_proxy' in os.environ}")
        
        # 调用工具
        output = tool.invoke(args)
        elapsed = time.time() - start_time
        print(f"✓ 工具'{tool_name}'执行完成，耗时: {elapsed:.2f}秒")
        
        # 生成响应
        if tool_name == "JD_search_general":
            return format_search_response(output)
        else:
            return {"role": "assistant", "content": f"已执行操作：{tool.name}\n结果：{str(output)[:200]}..."}
            
    except Exception as e:
        # 处理工具调用异常
        error_msg = f"工具'{tool_name}'调用失败：{str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return {"role": "assistant", "content": error_msg}

def format_search_response(output):
    """格式化搜索结果响应"""
    try:
        # 尝试解析JSON输出
        import json
        search_results = json.loads(output) if isinstance(output, str) else output
        
        if not isinstance(search_results, list) or not search_results:
            return {"role": "assistant", "content": f"搜索未返回有效结果: {str(output)[:200]}..."}
        
        # 检查是否有错误消息
        first_result = search_results[0]
        if "错误" in first_result.get("title", ""):
            return {"role": "assistant", "content": f"搜索时遇到问题: {first_result.get('title')}"}
        
        # 构建友好响应
        result_count = len(search_results)
        success_msg = f"已找到{result_count}个商品结果，为您整理如下："
        print(f"✅ {success_msg}")
        
        # 创建HTML表格
        html_table = "<table border='1' style='width:100%; border-collapse:collapse;'>"
        html_table += "<tr style='background-color:#f2f2f2;'><th>商品名称</th><th>价格</th><th>图片</th><th>购买链接</th></tr>"
        
        for product in search_results[:5]:  # 限制显示前5个结果
            title = product.get("title", "无标题")
            price = product.get("price", "价格未知")
            image_url = product.get("image_url", "#")
            purchase_link = product.get("purchase_link", "#")
            
            # 生成图片HTML标签 (小尺寸显示)
            img_html = f"<img src='{image_url}' width='100' alt='商品图片'/>" if image_url != "#" else "无图片"
            
            # 生成链接HTML标签
            link_html = f"<a href='{purchase_link}' target='_blank'>购买链接</a>" if purchase_link != "#" else "无链接"
            
            # 添加表格行
            html_table += f"<tr><td>{title}</td><td>{price}</td><td>{img_html}</td><td>{link_html}</td></tr>"
            
        html_table += "</table>"
        
        # 添加备用的Markdown表格（如果HTML不被支持）
        markdown_table = "| 商品名称 | 价格 | 购买链接 |\n| --- | --- | --- |\n"
        for product in search_results[:5]:
            title = product.get("title", "无标题")[:30] + "..." if len(product.get("title", "")) > 30 else product.get("title", "无标题")
            price = product.get("price", "价格未知")
            purchase_link = product.get("purchase_link", "#")
            link_text = f"[购买链接]({purchase_link})" if purchase_link != "#" else "无链接"
            markdown_table += f"| {title} | {price} | {link_text} |\n"
        
        # 组合响应
        result_msg = f"{success_msg}\n\n{html_table}\n\n如果表格显示不正确，请参考以下内容：\n\n{markdown_table}"
        print(f"📊 返回搜索结果: {len(search_results)}条")
        
        return {"role": "assistant", "content": result_msg}
        
    except Exception as e:
        # 如果解析失败，返回原始输出
        error_msg = f"已执行搜索操作，但结果解析失败: {str(e)}。原始结果: {str(output)[:200]}..."
        print(f"❌ {error_msg}")
        return {"role": "assistant", "content": error_msg}

llm_with_tools = llm.bind_tools(tools)

def maybe_route_to_tools(state: JD_QueryState) -> Literal["tools", "human", "__end__"]:
    """路由逻辑增加终止判断"""
    print(f"🔀 状态路由: 消息数量={len(state.get('messages', []))}, 完成状态={state.get('finished', False)}")
    
    # 检查完成状态
    if state.get("finished", False):
        print("🛑 检测到完成标记，结束流程")
        return "__end__"

    # 获取最新消息
    last_msg = state["messages"][-1]
    print(f"⬆️ 最新消息类型: {type(last_msg).__name__}")
    
    # 检测工具调用
    has_tool_calls = check_for_tool_calls(last_msg)
    
    # 确定路由
    route = "tools" if has_tool_calls else "human"
    print(f"🔀 路由决策: {route}")
    return route

def check_for_tool_calls(message) -> bool:
    """检查消息是否包含工具调用"""
    # 字典类型消息
    if isinstance(message, dict):
        has_tool_calls = "tool_calls" in message and message.get("tool_calls")
        tool_name = message.get("tool_calls", [{}])[0].get("name", "未知工具") if has_tool_calls else "无"
        print(f"📦 字典消息，包含工具调用: {has_tool_calls}" + (f", 工具: {tool_name}" if has_tool_calls else ""))
        return has_tool_calls
    
    # 对象类型消息
    tool_name = "未知"
    has_tool_calls = False
    
    # 1. 检查标准tool_calls属性
    if hasattr(message, "tool_calls") and message.tool_calls:
        has_tool_calls = True
        tool_name = get_tool_name_from_object(message.tool_calls[0])
        print(f"🤖 检测到tool_calls格式工具调用: {tool_name}")
        return True
    
    # 2. 检查OpenAI旧格式function_call
    if hasattr(message, "additional_kwargs"):
        add_kwargs = message.additional_kwargs
        if isinstance(add_kwargs, dict) and "function_call" in add_kwargs:
            func_call = add_kwargs["function_call"]
            if isinstance(func_call, dict) and "name" in func_call:
                has_tool_calls = True
                tool_name = func_call["name"]
                print(f"🤖 检测到function_call格式工具调用: {tool_name}")
                return True
    
    return False

def get_tool_name_from_object(tool_call_obj):
    """从工具调用对象获取工具名称"""
    if hasattr(tool_call_obj, "name"):
        return tool_call_obj.name
    if isinstance(tool_call_obj, dict) and "name" in tool_call_obj:
        return tool_call_obj["name"]
    return "未知工具"

# 在原有代码基础上修改状态图配置
graph_builder = StateGraph(JD_QueryState)

# 添加核心节点（保持终端对话逻辑）
graph_builder.add_node("human", human_node)
graph_builder.add_node("chatbot", chatbot_with_tools)
graph_builder.add_node("tools", tool_node)

# test_app.py 中调整状态图配置
graph_builder.add_conditional_edges(
    "chatbot",
    maybe_route_to_tools,
    {
        "tools": "tools",
        "human": "human",
        "__end__": END  # 新增终止路径
    }
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph_builder.add_edge(START, "chatbot")  # 初始化入口

graph_with_tools = graph_builder.compile()

mermaid_syntax = graph_with_tools.get_graph().draw_mermaid()
print(mermaid_syntax)


config = {"recursion_limit": 100}

user_msg = "你好，请自我介绍一下。"

# 将字符串转换为符合 JD_QueryState 的字典
# 修改main部分的初始状态
initial_state = JD_QueryState(
    messages=[{"role": "user", "content": user_msg}],
    query=[],
    finished=False
)

# 将直接导入和启动改为条件判断，避免循环导入
if __name__ == "__main__":
    from Gradio_UI import GradioUI
    GradioUI(graph_with_tools).launch(share=True)

async def jd_search_general(search_keyword, cookies_file):
    """异步执行京东商品搜索"""
    print(f"🔍 开始搜索京东商品: {search_keyword}")
    
    try:
        async with async_playwright() as playwright:
            # 初始化浏览器环境
            browser_context = await setup_browser(playwright)
            
            # 加载Cookie
            await load_cookies(browser_context, cookies_file)
            
            # 创建页面并导航到京东
            page = await browser_context.new_page()
            
            # 执行搜索流程
            search_result = await perform_search(
                page, 
                search_keyword, 
                browser_context
            )
            
            return search_result
            
    except Exception as e:
        error_msg = f"搜索过程中发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        return [{"title": error_msg, "price": "N/A"}]

async def setup_browser(playwright):
    """设置浏览器环境"""
    print("🌐 启动浏览器中...")
    browser = await playwright.chromium.launch(headless=True)
    return await browser.new_context()

async def load_cookies(browser_context, cookies_file):
    """加载Cookie文件"""
    try:
        with open(cookies_file, "r") as f:
            cookies = json.load(f)
            await browser_context.add_cookies(cookies)
        print(f"🍪 已加载Cookie文件: {cookies_file}")
    except Exception as e:
        print(f"⚠️ 读取Cookie文件时出错: {str(e)}")

async def perform_search(page, search_keyword, browser_context):
    """在京东执行搜索流程"""
    # 访问京东首页
    if not await navigate_to_jd(page):
        await browser_context.browser.close()
        return [{"title": "访问京东首页失败", "price": "N/A"}]
    
    # 执行搜索
    if not await execute_search_query(page, search_keyword):
        await browser_context.browser.close()
        return [{"title": f"搜索'{search_keyword}'失败", "price": "N/A"}]
    
    # 提取商品信息
    product_details = await extract_product_details(page, search_keyword)
    
    # 保存结果
    save_product_details(product_details)
    
    # 关闭浏览器
    await browser_context.browser.close()
    return product_details

async def navigate_to_jd(page):
    """导航到京东首页"""
    try:
        print("🌐 正在导航到京东首页...")
        await page.goto("https://www.jd.com", timeout=30000)
        print("✅ 京东首页加载完成")
        return True
    except Exception as e:
        error_msg = f"访问京东首页失败: {str(e)}"
        print(f"❌ {error_msg}")
        return False

async def execute_search_query(page, search_keyword):
    """执行搜索查询"""
    # 填充搜索框
    try:
        print("🔍 等待搜索框加载...")
        search_box_selector = "#key"
        await page.wait_for_selector(search_box_selector, timeout=10000)
        await page.fill(search_box_selector, search_keyword)
        print(f"✓ 已输入搜索关键词: {search_keyword}")
    except Exception as e:
        print(f"❌ 找不到或无法填充搜索框: {str(e)}")
        return False
    
    # 点击搜索并等待结果
    try:
        print("🔍 点击搜索按钮...")
        search_button_selector = ".button"
        await page.click(search_button_selector)
        print("⏳ 等待搜索结果加载...")
        await page.wait_for_selector(".gl-item", timeout=15000)
        print("✅ 搜索结果已加载完成")
        return True
    except Exception as e:
        print(f"❌ 搜索结果加载失败: {str(e)}")
        return False

async def extract_product_details(page, search_keyword):
    """提取商品详情"""
    product_details = []
    
    try:
        # 获取商品列表
        print("🔍 获取商品列表...")
        product_items = await page.query_selector_all(".gl-item")
        
        # 只处理前3个商品
        items_to_process = product_items[:min(3, len(product_items))]
        print(f"✓ 找到{len(product_items)}个商品，将处理前{len(items_to_process)}个")
        
        # 无商品情况处理
        if not items_to_process:
            print("⚠️ 未找到任何商品")
            return [{"title": f"没有找到与'{search_keyword}'相关的商品", "price": "N/A"}]
        
        # 处理每个商品
        for index, item in enumerate(items_to_process):
            print(f"🔍 处理商品 {index + 1}/{len(items_to_process)}...")
            await asyncio.sleep(random.uniform(3, 10))
            
            # 提取商品信息
            product_info = await process_product_item(item, index)
            product_details.append(product_info)
            
        # 确保至少有一个结果
        if not product_details:
            return [{"title": f"无法提取'{search_keyword}'的商品信息", "price": "N/A"}]
            
        return product_details
        
    except Exception as e:
        error_msg = f"提取商品信息时出错: {str(e)}"
        print(f"❌ {error_msg}")
        return [{"title": error_msg, "price": "N/A"}]

async def process_product_item(item, index):
    """处理单个商品项"""
    try:
        # 提取商品标题
        title_element = await item.query_selector(".p-name a")
        title = await title_element.inner_text() if title_element else "标题获取失败"

        # 获取商品的购买链接
        purchase_link = await title_element.get_attribute("href") if title_element else "#"
        purchase_link = f"https:{purchase_link}" if purchase_link.startswith("//") else purchase_link

        # 提取商品价格
        price_element = await item.query_selector(".p-price strong i")
        price = await price_element.inner_text() if price_element else "价格获取失败"

        # 提取商品图片链接
        image_element = await item.query_selector(".p-img img")
        image_url = await image_element.get_attribute("src") if image_element else "#"
        image_url = f"https:{image_url}" if image_url and image_url.startswith("//") else image_url
        
        # 不在这里直接调用图像处理函数，避免同步/异步混用问题
        # 只保存图片URL，后续可以使用tool再处理
        image_text = "图片内容将在查看时提取"
        
        # 构建商品信息
        product_info = {
            "title": title.strip(),
            "price": price.strip(),
            "image_url": image_url.strip(),
            "image_text": image_text.strip(),
            "purchase_link": purchase_link.strip()
        }
        
        print(f"✓ 商品 {index + 1}: {title.strip()[:30]}..., 价格: {price.strip()}")
        return product_info
        
    except Exception as e:
        error_msg = f"处理商品{index+1}时出错: {str(e)}"
        print(f"⚠️ {error_msg}")
        return {
            "title": f"商品{index+1}信息提取失败: {str(e)}",
            "price": "N/A",
            "image_url": "#",
            "image_text": '#',
            "purchase_link": "#"
        }

def save_product_details(product_details):
    """保存商品详情到JSON文件"""
    try:
        output_file = "jd_product_details.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(product_details, f, ensure_ascii=False, indent=4)
        print(f"✅ 商品详细信息已保存到 {output_file}")
    except Exception as e:
        print(f"⚠️ 保存JSON文件失败: {str(e)}")
