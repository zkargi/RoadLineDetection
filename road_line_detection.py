import cv2
import numpy as np

def region_of_interest(image, vertices):
    # Giriş görüntüsüyle aynı boyutlarda boş bir maske oluşturun
    mask = np.zeros_like(image)
    
    # Maske için renk tanımlayın
    maske_rengi = 255
    
    # Maskeyi, vertekslerle tanımlanan çokgenle doldurun
    cv2.fillPoly(mask, vertices, maske_rengi)
    
    # Maskeyi görüntüye bitwise AND işlemiyle uygulayın
    masked_image = cv2.bitwise_and(image, mask)
    
    # Maskeli görüntüyü döndürün
    return masked_image

def draw_lines(image, lines):
    # Giriş görüntüsünün bir kopyasını oluşturun
    image = np.copy(image)
    
    # Giriş görüntüsüyle aynı boyutlarda boş bir görüntü oluşturun
    blank_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    # Herhangi bir çizgi tespit edilip edilmediğini kontrol edin
    if lines is not None:
        # Çizgiler üzerinde yineleme yapın ve bunları boş görüntüye çizin
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)
            # İlk çizgiyi çizdikten sonra durmak için sonraki satırı uncomment yapabilirsiniz
            # break

    # Çizgileri, bazı şeffaflıklarla orijinal görüntünün üzerine bindirin
    image = cv2.addWeighted(image, 0.8, blank_image, 1, 0.0)
    
    # Çizgiler çizilmiş görüntüyü döndürün
    return image

def process(image):
    # Giriş görüntüsünün boyutlarını alın
    height, width = image.shape[0], image.shape[1]
    
    # İlgi bölgesi için verteksleri tanımlayın
    region_of_interest_verteksleri = [
        (0, height), 
        (width / 2, height / 2), 
        (width, height)
    ]
    
    # Görüntüyü gri tonlamaya çevirin
    gri_goruntu = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gri tonlamalı görüntüye Canny kenar algılamayı uygulayın
    canny_goruntu = cv2.Canny(gri_goruntu, 250, 120)
    
    # Görüntüyü ilgi bölgesine kırpın
    kirpilmis_goruntu = region_of_interest(canny_goruntu, np.array([region_of_interest_verteksleri], np.int32))
    
    # Hough transformunu kullanarak kırpılmış görüntüde çizgileri tespit edin
    cizgiler = cv2.HoughLinesP(
        kirpilmis_goruntu, 
        rho=2, 
        theta=np.pi/180, 
        threshold=200, 
        lines=np.array([]), 
        minLineLength=150, 
        maxLineGap=4
    )
    
    # Tespit edilen çizgileri orijinal görüntüye çizin
    cizgili_goruntu = draw_lines(image, cizgiler)
    
    # İşlenmiş görüntüyü döndürün
    return cizgili_goruntu

# Video dosyasını açın
cap = cv2.VideoCapture("video2.mp4")

while True:
    # Videodan bir kare okuyun
    success, img = cap.read()
    
    # Kerenin başarılı bir şekilde okunup okunmadığını kontrol edin
    if not success:
        break
    
    # Çizgileri tespit edip çizecek şekilde kareyi işleyin
    img = process(img)
    
    # İşlenmiş kareyi görüntüleyin
    cv2.imshow("img", img)
    
    # 20 milisaniye bekleyin ve bir tuş basımı olup olmadığını kontrol edin
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Video yakalama nesnesini serbest bırakın ve tüm OpenCV pencerelerini kapatın
cap.release()
cv2.destroyAllWindows()
