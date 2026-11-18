# Chrome task agent

Give the agent a starting URL and describe the result you want. It opens Google Chrome, reads the visible page, and chooses one action at a time. It reports the observed result or the reason it stopped and writes a run log under `logs/`.

## Set up on Windows

Install Python 3.10 or newer and Google Chrome. Run these commands in PowerShell from this folder.

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `python` opens the Microsoft Store or fails to start, use the full path to your installed `python.exe` in the first command. If `.env` is missing, copy `.env.example` to `.env`. Paste your [OpenAI API key](https://platform.openai.com/api-keys) into `OPENAI_API_KEY` in `.env`. The default model is `gpt-4.1-mini`. The key needs API access to the selected model.

## Run a task

```powershell
& .\.venv\Scripts\python.exe agent.py --url 'https://example.com' --task 'Read the main heading and tell me what this page is for.'
```

Run `agent.py` without options for interactive URL and task prompts. Use `--login` to open the page and sign in manually before the agent begins. The Chrome profile lives in `.chrome-profile/` and retains logins between runs. Run one agent at a time with this profile.

The agent asks for approval before clicking, typing, or sending keys. Check the visible page and the proposed action, then enter `y` to proceed. If the viewport changes while waiting, the agent will take a fresh look. Press `Ctrl+C` to stop. The agent closes its Chrome window at the end of the run.

The terminal shows the log path for each run. Use `--steps 30` to change the default 20-decision limit. Exit code `0` means the model reported a verified result, `1` means the run stopped short or failed, `2` means setup or usage needs attention, and `130` means the terminal interrupted the run or closed input.

The task text and visible browser screenshots are sent to the OpenAI API, which can incur usage charges. Screenshots are held in memory. The log records the initial host, action types, and result or blocker, so the final log message may contain details from the page. Use `--login` to enter credentials directly in Chrome. The agent works inside web pages and can stop on native dialogs, downloads, CAPTCHAs, and highly dynamic pages.

[OpenAI API quickstart](https://developers.openai.com/api/docs/quickstart) · [Playwright Chrome support](https://playwright.dev/python/docs/browsers#google-chrome--microsoft-edge)
