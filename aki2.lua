


if not getgenv().__panda_elc_a164d84c82ba4a4f then
getgenv().__panda_elc_a164d84c82ba4a4f = true
local jsonEncode = game:GetService("HttpService").JSONEncode
task.spawn(function()
    while task.wait(10) do
        local ok, err = pcall(function()
            request({
                Url = "https://vss.pandadevelopment.net/execute_information",
                Method = "POST",
                Headers = {["Content-Type"] = "application/json"},
                Body = jsonEncode(game:GetService("HttpService"), {
                    slug_id = "a164d84c82ba4a4f",
                    executor_name = identifyexecutor and identifyexecutor() or "Unknown",
                    hardware_id = gethwid and gethwid() or "",
                    job_id = tostring(game.JobId),
                    place_id = tostring(game.PlaceId),
                }),
            })
        end)
        if not ok then
            warn("[Panda VSS] Execute information failed:", err)
        end
    end
end)
end



local RunService = game:GetService("RunService")

local count = 0
local conn

conn = RunService.Heartbeat:Connect(function()
    count += 1
    if count >= 10 then
        conn:Disconnect()
    end
end)

while count < 10 do
    RunService.Heartbeat:Wait()
end

if count < 10 then
    error("Eror")
    return
end


-- new key system
local UI = loadstring(game:HttpGet("https://pastebin.com/raw/Rwva1iVH"))()
local HttpService = game:GetService("HttpService")
local KEY_FILE = "RcHub.json"

local BaseURL = "https://new.pandadevelopment.net/api/v1"
local Client_ServiceID = "rexycode"

local GameLoaders = {
    [9186719164]  = "https://vss.pandadevelopment.net/virtual/file/5d89639a6e4d471e",
    [119987266683883] = "https://vss.pandadevelopment.net/virtual/file/af185d5c2e424cbc",
}

local function getHardwareId()
    local success, hwid = pcall(gethwid)
    if success and hwid then
        return hwid
    end
    local RbxAnalyticsService = game:GetService("RbxAnalyticsService")
    local clientId = tostring(RbxAnalyticsService:GetClientId())
    return clientId:gsub("-", "")
end

local function makeRequest(endpoint, body)
    local url = BaseURL .. endpoint
    local jsonBody = HttpService:JSONEncode(body)

    local response = request({
        Url = url,
        Method = "POST",
        Headers = {
            ["Content-Type"] = "application/json"
        },
        Body = jsonBody
    })

    if response and response.Body then
        return HttpService:JSONDecode(response.Body)
    end

    return nil
end

local function Validate(key, premiumCheck)
    local hwid = getHardwareId()

    local result = makeRequest("/keys/validate", {
        ServiceID = Client_ServiceID,
        HWID = hwid,
        Key = key
    })

    if not result then
        return {
            success = false,
            isPremium = false
        }
    end

    local isAuthenticated = result.Authenticated_Status == "Success"
    local isPremium = result.Key_Premium or false

    local isValid = isAuthenticated

    if premiumCheck and isAuthenticated and not isPremium then
        isValid = false
    end

    return {
        success = isValid,
        isPremium = isPremium
    }
end

local function GetKeyURL()
    local hwid = getHardwareId()
    return "https://new.pandadevelopment.net/getkey/" .. Client_ServiceID .. "?hwid=" .. hwid
end

function LoadData()
    local loader = GameLoaders[game.GameId]
    if loader then
        loadstring(game:HttpGet(loader))()
    else
        game:GetService("Players").LocalPlayer:Kick("Game not supported")
    end
end

local function SaveKey(key)
    if writefile then
        writefile(KEY_FILE, HttpService:JSONEncode({ key = key }))
    end
end

local function LoadKey()
    if readfile and isfile and isfile(KEY_FILE) then
        local success, result = pcall(function()
            return HttpService:JSONDecode(readfile(KEY_FILE))
        end)
        if success and result and result.key then
            return result.key
        end
    end
    return nil
end

local function DeleteKey()
    if delfile and isfile and isfile(KEY_FILE) then
        delfile(KEY_FILE)
    end
end

function n(txt)
    UI:Notify({
        Title = "RcHub - Key System",
        Message = txt,
        Duration = 3
    })
end

local savedKey = LoadKey()

if savedKey then
    local result = Validate(savedKey, false)
    if result.success then
        if result.isPremium then
            n("Script Load!")
            n("Status: Premium")
            LoadData()
        else
            n("Script Load!")
            n("Status: Normal")
            LoadData()
        end
    else
        DeleteKey()
    end
end

if not savedKey or not Validate(savedKey, false).success then
    n("Invalid Key, Please get New Key.")

    local Window = UI:CreateWindow({
        Title = "RC HUB | KEY SYSTEM",
        Subtitle = "1 Checkpoint · 24 Hours Key",
        Width = 400,
        Height = 250
    })

    Window:CreateInput({
        Placeholder = "Enter your key...",
        Callback = function(text)
            local result = Validate(text, false)
            if result.success then
                Window:Destroy()
                SaveKey(text)

                if result.isPremium then
                    n("Script Load!")
                    n("Status: Premium")
                else
                    n("Script Load!")
                    n("Status: Normal")
                end

                LoadData()
            else
                n("Invalid Key, Please get New Key.")
            end
        end
    })

    Window:CreateButton({
        Text = "Get Key",
        Callback = function()
            setclipboard(GetKeyURL())
            Window:Notify({
                Title = "Get Key",
                Message = "Link Already Copied, Paste On Browser.",
                Duration = 3
            })
        end
    })

    Window:CreateButton({
        Text = "Discord",
        Callback = function()
            setclipboard("https://discord.gg/CpcnQ9DHng")
            Window:Notify({
                Title = "Discord",
                Message = "Discord link Copied.",
                Duration = 3
            })
        end
    })
end
