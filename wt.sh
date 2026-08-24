. /home/parallels/Desktop/charsiu/scripts/charsiu-tui.sh
echo "CTUI=$CTUI" > /tmp/claude-1000/-home-parallels-Desktop-linux-rk3576-npu/110b1058-9c12-47c9-a4c1-42e711d8d410/scratchpad/wt.out
sel=$(ui_menu "pick one" alpha "the first" beta "the second" gamma "the third")
echo "menu rc=$? sel='$sel'" >> /tmp/claude-1000/-home-parallels-Desktop-linux-rk3576-npu/110b1058-9c12-47c9-a4c1-42e711d8d410/scratchpad/wt.out
v=$(ui_input "threads" "4")
echo "input rc=$? v='$v'" >> /tmp/claude-1000/-home-parallels-Desktop-linux-rk3576-npu/110b1058-9c12-47c9-a4c1-42e711d8d410/scratchpad/wt.out
if ui_yesno "proceed?"; then echo "yesno=YES" >> /tmp/claude-1000/-home-parallels-Desktop-linux-rk3576-npu/110b1058-9c12-47c9-a4c1-42e711d8d410/scratchpad/wt.out
else echo "yesno=NO" >> /tmp/claude-1000/-home-parallels-Desktop-linux-rk3576-npu/110b1058-9c12-47c9-a4c1-42e711d8d410/scratchpad/wt.out; fi
