from playwright.async_api import async_playwright
import json, asyncio, nest_asyncio

async def save_and_verify_jd_cookies():
    """
    This function uses Playwright to automate the process of logging into JD.com,
    saving the cookies to a file, and verifying the login status.
    """
    async with async_playwright() as playwright:
        # start a new browser instance
        # set headless=False to interact with the browser
        browser = await playwright.chromium.launch(headless=False)  
        context = await browser.new_context()
        page = await context.new_page()

        # navigate to JD login page
        await page.goto("https://passport.jd.com/new/login.aspx")
        print("Please manully scan the QR_Code...")

        # awiat for user to login
        # This is a blocking call, it will wait for 30 seconds
        # You can adjust the time as needed
        await page.wait_for_timeout(30000)  

        # check if login was successful
        current_url = page.url
        if "passport.jd.com" in current_url:
            print("Login failed, please try again.")
            await browser.close()
            return

        # save cookies to jd_cookies.json
        cookies = await context.cookies()
        cookies_file = "jd_cookies.json"
        with open(cookies_file, "w", encoding="utf-8") as f:
            json.dump(cookies, f, ensure_ascii=False, indent=4)
            print(f"Cookie saved to {cookies_file}")

        # verify if the cookies are valid
        print("Verifying cookies...")
        new_context = await browser.new_context()
        await new_context.add_cookies(cookies)
        new_page = await new_context.new_page()
        await new_page.goto("https://www.jd.com")

        # check if the login was successful
        if "jd.com" in new_page.url:
            print("Cookie verification successful, you are logged in.")
        else:
            print("Cookie verification failed, please check your ternimal output.")

        # close the browser
        await browser.close()

# run the function
nest_asyncio.apply()
asyncio.run(save_and_verify_jd_cookies())
