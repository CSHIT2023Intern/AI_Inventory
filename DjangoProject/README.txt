# python 版本 3.7.13
#權重檔檔案過大，已共用在雲端硬碟(20220817.pt)
#權重檔(.pt)需放在static資料夾中
#更換權重檔需修改detect.py-->line 12 以及 views.py -->line 11

#藥品標記壓縮檔共用在雲端硬碟(藥品標記檔.rar)

#初始參數設定為 : weights='static/20220817.pt',img_size=640, conf_thres=0.7,iou_thres=0.7, device='cpu'  -->  於detect.py  line 15~17
#可根據日後訓練成果調整

#參考來源 : https://blog.csdn.net/weixin_51331359/article/details/126012620 (爆改YOLOV7的detect.py制作成API接口供其他python程序調用)
python manage.py runserver

(2023/08/15)
#FastAPI
-->conda activate yolov7
-->cd C:\Users\user\Desktop\中山醫醫資\大三下\實習\112_intern\AI_Inventory\DjangoProject\MyFirstDjango
-->uvicorn app:app --reload

使用fastAPI介面
http://localhost:8000/docs
輸入image的base64字串
點選excute
輸出 vial_count、ampoule_count、total_count、result_image_base64

使用postman
post
http://localhost:8000/detect
body選x-www-form-urlencoded
key:image_data
value:(圖片的base64字串)
點選sand即可
輸出 vial_count、ampoule_count、total_count、result_image_base64
/////
#MangoDB
-->conda activate yolov7
-->cd C:\Users\user\Desktop\中山醫醫資\大三下\實習\112_intern\AI_Inventory\DjangoProject\MyFirstDjango
-->python manage.py runserver
http://localhost:8000/Index
進入後即可操作上傳圖片和選擇照片