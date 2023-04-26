import cv2


#Định nghĩa hàm getFaceBox dau vao  faceNet và khung hình  frame:
def getFaceBox(faceNet, frame):
    frameHeight = frame.shape[0] 
    frameWidth = frame.shape[1]
    
# Sử dụng hàm cv2.dnn.blobFromImage để chuyển đổi khung hình thành định dạng dữ liệu đầu vào cho mô hình faceNet. 
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    
# faceNet.setInput được sử dụng để gán giá trị  blob cho đầu vào của mô hình faceNet.
    faceNet.setInput(blob)
# Sử dụng faceNet.forward() để phát hiện khuôn mặt trong khung hình.
    detection = faceNet.forward()
    faceBoxes = []
    
#vòng lặp để duyệt qua các phát hiện được tìm thấy trong biến detection. Vì mỗi phát hiện được biểu diễn bằng một hàng trong ma trận detection
#nên chúng ta sử dụng detection.shape[2] để xác định số lượng phát hiện``
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        
#Lấy các giá trị tọa độ của các khuôn mặt được phát hiện với độ tin cậy lớn hơn 0,7 và thêm chúng vào danh sách faceBoxes.
        if confidence > 0.7:
            
# Vẽ hộp giới hạn xung quanh các khuôn mặt được phát hiện trên khung hình frame

# detection là một mảng chứa các thông tin về khuôn mặt được phát hiện trong khung hình, được trích xuất từ đầu ra của mô hình faceNet.
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
#cv2.rectangle: vẽ hop tren khung hinh
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
# và Trả về khung hình frame cùng với danh sách faceBoxes chứa tọa độ của các khuôn mặt được phát hiện.
    return frame, faceBoxes


#định nghĩa các đường dẫn đến các tệp mô hình nhận dạng khuôn mặt

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

#sử dụng cv2.dnn.readNet để đọc tệp mô hình và tạo các đối tượng mô hình tương ứng.
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)



#(78.4263377603, 87.7689143744, 114.895847746) là giá trị trung bình của các kênh màu RGB trong tập dữ liệu 

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

video = cv2.VideoCapture(0)
padding = 20

while True:
    hasFrame, vidFrame = video.read()

    if not hasFrame:
        cv2.waitKey()
        break

    frame, faceBoxes = getFaceBox(faceNet, vidFrame)

    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
               max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]


#Hàm cv2.dnn.blobFromImage để chuyển đổi khung hình thành định dạng dữ liệu đầu vào cho mạng neural.      
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        genderNet.setInput(blob) #Tạo blob  từ khung hình frame sử dụng phương thức cv2.dnn.blobFromImage để chuẩn bị dữ liệu cho mạng neural.
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]

        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]
        labelGender = "{}".format("Gender : " + gender)
        labelAge = "{}".format("Age : " + age + "Years")
        
        
#cv2.putText:sử dụng để vẽ các văn bản lên khung hình 
        cv2.putText(frame, labelGender, (faceBox[0], faceBox[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, labelAge, (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Age-Gender Detector", frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
