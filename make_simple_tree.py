import os

class address():

    def __init__(self , address , length):
        address = input("경로를 입력하여 주세요")
        address_list = address.split('\\')
        self.address = address_list
        print("현재 주소는",self.address)
        length = len(self.address)
        self.length = length

    def printeach(self):
        i = 1
        viradd = ''
        length1 = self.length
        length2 = self.length
        print("총 길이는", self.length)
        while (length1):
            viradd = viradd + self.address[length2 - length1] + '\\\\'
            length1 = length1 - 1
            file_list = os.listdir(viradd)
            print(file_list)
            f = open("C://Users//jbin7_000//Desktop//test.txt",'at')
            f.write(viradd)
            for files in file_list:
                f.write(files)
                f.write('\n')
            f.close()

mycom = address('?' , 1)
mycom.printeach()


