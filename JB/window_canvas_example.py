from tkinter import *
root = Tk()
canvas = Canvas(root , width = 300 , height = 300)
#캔버스 이름은 root , 크기 조정
def draw(event):
    global x0, y0
    canvas.create_line(x0, y0, event.x, event.y)
    x0 , y0 = event.x , event.y
    #마우스가 움직임에 따라 점들이 찍힙니다.
def down(event):
    global x0 , y0
    x0, y0 = event.x, event.y
    #마우스로 클릭할때 점의 위치를 잡습니다.
def up(event):
    global x0 , y0
    if(x0 , y0) == (event.x , event.y):
        canvas.create_line(x0 , y0 , x0+1 , y0 +1)
        #마우스로 클릭하고 띄었을 때 그 자리에 점이 생깁니다.
def clear(event):
    canvas.delete("all")
    #캔버스를 모두 지워버립니다.


canvas.bind("<B1-Motion>", draw) # 마우스의 움직임 체킹
canvas.bind("<Button-1>", down) # 마우스의 클릭 체킹
canvas.bind("<ButtonRelease-1>", up) # 클릭상태 해제 체킹
canvas.bind("<Button-3>", clear) # Button123은 각각 Left, middle , right 입니다.
canvas.pack() # 캔버스 돌리기
root.mainloop()
