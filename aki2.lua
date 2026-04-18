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
    error("Error")
    return
end

-- [ĐÃ BYPASS: XÓA BỎ HỆ THỐNG KEY]
local UI = loadstring(game:HttpGet("https://pastebin.com/raw/Rwva1iVH"))()

local GameLoaders = {
    [9186719164]  = "https://vss.pandadevelopment.net/virtual/file/5d89639a6e4d471e",
    [119987266683883] = "https://vss.pandadevelopment.net/virtual/file/af185d5c2e424cbc",
}

function n(txt)
    UI:Notify({
        Title = "RcHub - Bypassed",
        Message = txt,
        Duration = 3
    })
end

function LoadData()
    local loader = GameLoaders[game.GameId]
    if loader then
        local success, err = pcall(function()
            loadstring(game:HttpGet(loader))()
        end)
        if not success then
            warn("Lỗi tải Server Payload: " .. tostring(err))
            n("Server từ chối kết nối (Có thể do chặn HWID)!")
        end
    else
        game:GetService("Players").LocalPlayer:Kick("Game not supported")
    end
end

-- Chạy trực tiếp Script chính
n("Script Load!")
n("Status: Premium Unlocked")
LoadData()
