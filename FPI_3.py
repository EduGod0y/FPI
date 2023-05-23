import cv2
import pyautogui
import numpy as np

camera = 0
cap = cv2.VideoCapture(camera)

if not cap.isOpened():
    exit()

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    blur = cv2.GaussianBlur(frame, (11, 11), 0)
    #------------------------------------------------------------------------------------------------------------------------------
    canny = cv2.Canny(frame, 10, 70)
    #------------------------------------------------------------------------------------------------------------------------------
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    src = cv2.GaussianBlur(frame, (3, 3), 0)
    
    
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    #------------------------------------------------------------------------------------------------------------------------------
    ajust = cv2.addWeighted(frame, 0.8, frame, 0, 4)

    #-------------------------------------------------------------------------------------------------------------------------------
    	
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #-------------------------------------------------------------------------------------------------------------------------------
    

    resize = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
    #------------------------------------------------------------------------------------------------------------------------------

    rotate = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    #------------------------------------------------------------------------------------------------------------------------------

    flipped_h = cv2.flip(frame, 1)
    #------------------------------------------------------------------------------------------------------------------------------

    flipped_v = cv2.flip(frame, -1)
    #------------------------------------------------------------------------------------------------------------------------------
    cv2.imshow("Imagem original :)", frame)
    #cv2.imshow('Gasussian blur ', blur)
    #cv2.imshow('Canny (edge detector)', canny)
    #cv2.imshow('Gradiente', grad)
    #cv2.imshow('Contraste e Brilho ajustados', ajust)
    #cv2.imshow('Imagem em Tons de Cinza', gray)
    #cv2.imshow('Negativo da Imagem', 255 - frame)
    #cv2.imshow('Imagem com metade dos pixels', resize)
    #cv2.imshow('Imagem rotacionda 90°', rotate)
    #cv2.imshow('Imagem invertida horizontalmente', flipped_h)
    cv2.imshow('Imagem invertida verticalmente', flipped_v)
    if cv2.waitKey(1) == 27:
        break
cap.release()

cv2.destroyAllWindows()


