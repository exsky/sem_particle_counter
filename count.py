import cv2

# 從 image 目錄讀取彩色圖檔 ori_1.jpg
# cv2.imread 的第二個參數帶 0 時，會將圖片轉換成灰階
# 然後派到變數 gray1
gray1 = cv2.imread("image/ori_1.jpg", 0)

# 將 gray1 變數(圖片物件) 寫到檔案
# 其路徑為 目錄 image/
# 並且將檔名命名為 gray_1.jpg
cv2.imwrite("image/gray_1.jpg", gray1)

# Binarization 二階化
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
# 先閱讀上面連結的文件，依照所需進行調整
# 設定各項閥值，將圖片進行二階化轉換
# 以下範例為，輸入灰階圖檔 gray1，並且以灰色深度 135 做為分界
# 顏色深度大於 135 的值，變成 255
# 深度小於等於 135 的像素，變成 0
# 第四個參數則是採取的轉換模式 (請參考官方文件)
# 因此 cv2.THRESH_BINARY 讓輸出的圖片最終只有 純黑與純白
ret_value1, thresed_1 = cv2.threshold(gray1, 135, 255, cv2.THRESH_BINARY)
cv2.imwrite("image/gb1_1.jpg", thresed_1)

# 待修正問題1：破碎/中空填滿
# 參考 https://stackoverflow.com/questions/10316057/filling-holes-inside-a-binary-object
ret_value1, thresed_z_1 = cv2.threshold(gray1, 135, 255, cv2.THRESH_TOZERO)
cv2.imwrite("image/gb1_z_1.jpg", thresed_z_1)

# 待修正問題2：重疊/相鄰粒子

# findcontours
# 將圖片中的黑點位置標記出來 / 將座標存到 cnts 串列中
# cnts = cv2.findContours(threshed_1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
cnts = cv2.findContours(thresed_1,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)[-2]


# 自創一個函式，叫做 countBall
# 並制定使用這個函式時，需帶入兩個參數 s1, s2
# 兩個參數分別代表 遮罩的大小 (最小和最大的範圍)
# 抓取大小介於 s1 ~ s2 的區域數量
def countBall(s1, s2):
    xcnts = []
    for cnt in cnts:
        if s1 < cv2.contourArea(cnt) <s2:
            xcnts.append(cnt)
    return xcnts

# 觀察圖片，因為粗估大部分的粒徑佔 30 ~ 38 像素
countBall(32, 38)

# 由於 問題2 尚未解決，導致相鄰的兩個粒子會被當成 1 顆計算 / 亦或是超出 s2 上限而未被計算到
