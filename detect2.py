import cv2
import numpy as np
import time

def ilgi_bolgesi(image, vertices):
    mask = np.zeros_like(image)
    maske_rengi = 255
    cv2.fillPoly(mask, vertices, maske_rengi)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def cizgileri_ciz(image, lines, renk=(0, 255, 0), kalinlik=10):
    image = np.copy(image)
    blank_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(blank_image, (x1, y1), (x2, y2), renk, kalinlik)
    image = cv2.addWeighted(image, 0.8, blank_image, 1, 0.0)
    return image

def isleme(image):
    height, width = image.shape[0], image.shape[1]
    ilgi_bolgesi_verteksleri = [
        (0, height), 
        (width / 2, height / 2), 
        (width, height)
    ]
    
    gri_goruntu = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur uygula
    bulanik_goruntu = cv2.GaussianBlur(gri_goruntu, (5, 5), 0)
    
    canny_goruntu = cv2.Canny(bulanik_goruntu, 250, 120)
    kirpilmis_goruntu = ilgi_bolgesi(canny_goruntu, np.array([ilgi_bolgesi_verteksleri], np.int32))
    
    cizgiler = cv2.HoughLinesP(
        kirpilmis_goruntu, 
        rho=2, 
        theta=np.pi/180, 
        threshold=200, 
        lines=np.array([]), 
        minLineLength=150, 
        maxLineGap=4
    )
    
    cizgili_goruntu = cizgileri_ciz(image, cizgiler, renk=(0, 255, 0), kalinlik=5)
    return cizgili_goruntu

cap = cv2.VideoCapture("video2.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
start_time = time.time()

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = isleme(img)
    
    # FPS bilgisini görüntüye ekle
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps_display = frame_count / elapsed_time
    cv2.putText(img, f"FPS: {fps_display:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("img", img)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
