
from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *
import decode


width =100
height = 100
center = height//2
white = (255, 255, 255)
green = (0,128,0)
count = 1



def save():
    global count
    filename = "image_"+str(count)+"_1.png"
    count = count + 1
    image1.save(filename)
    decode.imageprepare('C://Users//jbin7_000//Desktop//ten//'+filename)



def paint(event):
    # python_green = "#476042"
    x1, y1 = (event.x ), (event.y )
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=5)
    draw.line([x1, y1, x2, y2],fill="black",width=5)

def clear():
    cv.delete("all")
    global image1 , draw
    del image1
    image1 = PIL.Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image1)

    #캔버스를 모두 지워버립니다.

root = Tk()

# Tkinter create a canvas to draw on
cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()

# PIL create an empty image and draw object to draw on
# memory only, not visible
image1 = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)

# do the Tkinter canvas drawings (visible)
# cv.create_line([0, center, width, center], fill='green')

cv.pack(expand=NO, fill=BOTH)
#공백 채우는 옵션입니다.

cv.bind("<B1-Motion>", paint)

# do the PIL image/draw (in memory) drawings
# draw.line([0, center, width, center], green)

# PIL image can be saved as .png .jpg .gif or .bmp file (among others)
# filename = "my_drawing.png"
# image1.save(filename)
button=Button(text="save",command=save)
button1 = Button(text = 'clear' , command = clear )
button.pack()
button1.pack()
root.mainloop()

