set dFolder to "~/Desktop/screencapture/"

do shell script ("mkdir -p " & dFolder)

set i to 0
repeat 1300 times
    -- 캡처할 영역의 좌표와 크기를 지정하여 스크린캡처
    do shell script ("screencapture -R 9,69,701,910 " & dFolder & "frame-" & i & ".png")
    delay 1 -- 1초 대기

    -- 오른쪽 화살표 키보드 이벤트 보내기
    tell application "System Events"
        key code 124 -- 오른쪽 화살표 키 코드
    end tell

    delay 1 -- 1초 대기
    set i to i + 1
end repeat