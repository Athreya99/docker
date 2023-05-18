import cv2
import dlib
#dependencies--dlib,cv2,cmake,points file
# Step 1: Load the shape predictor model
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Step 2: Load the input image
image = cv2.imread('image.jpeg')

# Step 3: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 4: Detect faces in the image
detector = dlib.get_frontal_face_detector()
faces = detector(gray)

# Step 5: Iterate over the detected faces and extract the facial landmarks
for face in faces:
    landmarks = predictor(gray, face)
    points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        points.append((x, y))

    # Step 6: Compute the Delaunay triangulation on the extracted landmarks
    rect = (0, 0, image.shape[1], image.shape[0])
    subdiv = cv2.Subdiv2D(rect)
    for point in points:
        subdiv.insert(point)
    triangle_list = subdiv.getTriangleList()

    # Step 7: Draw the Delaunay triangles on the image
    for t in triangle_list:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        cv2.line(image, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(image, pt2, pt3, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(image, pt3, pt1, (0, 255, 0), 1, cv2.LINE_AA)

# Step 8: Display the image with Delaunay triangles
cv2.imshow('Delaunay Triangulation', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
