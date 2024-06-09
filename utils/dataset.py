import os
import pickle
from face_recognition import load_image_file,face_locations,face_encodings
from cv2 import cvtColor, COLOR_BGR2RGB
class Dataset():    
    def __init__(self,images_folder_path:str=None):
        if os.path.isdir(images_folder_path):
            self.__images_folder_path = images_folder_path
        else:
            raise Exception("Check the path betchhh!!")
        

    def EncodeImages(self):
        self.knownEncodings = []
        self.knownNames = []
        i = 0
        
        for image_folder in os.listdir(self.__images_folder_path):

            for image_file in os.listdir(self.__images_folder_path+'/'+image_folder):
                print(f"{image_file} image processing...")
                image = load_image_file(self.__images_folder_path+'/'+image_folder+'/'+image_file)
                image = cvtColor(image,COLOR_BGR2RGB)
                name = image_folder
                boxes = face_locations(image)
                encodings = face_encodings(image,boxes)
                i+=1
                if encodings is not None:
                    for encoding in encodings:
                        self.knownEncodings.append(encoding)
                        self.knownNames.append(name)
        
        return self.knownEncodings,self.knownNames   
        
    def SerializeImages(self,encoding_path:str="./encodings/version_1.pickle"):
        print("\nSerializing encodings faces...\n")
        encodings,names = self.EncodeImages()
        data = {"encodings":encodings,"names":names}
        f = open(encoding_path,"wb")
        f.write(pickle.dumps(data))
        f.close()

    def DeSerializeImages(self,pickle_path:str):
        print("\nLoading encodings...\n")
        data = pickle.loads(open(pickle_path, "rb").read())
        return data
        



