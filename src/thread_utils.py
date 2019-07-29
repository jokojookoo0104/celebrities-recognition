import sys
sys.path.append('../')
from triplet import *
from data import *
from src.preprocess import *
from src.model import *
from src.pretrain_model import *
from src.face_detect import face_detect
from src.config import *
from utils import *
from detector import detect_faces
from PIL import Image
from align import *
from visualization import show_results

import threading
import zipfile
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import Graph, Session


#root = './../database/'
#UPLOAD_FOLDER = './database'
#IMG_FOLDER= './face_ss'
global_lock = threading.Lock()
model = face_detect()
#update class in dataset
class UpdateData(threading.Thread):
    def __init__(self, datasetid,class_name, images):
        threading.Thread.__init__(self)
        self.datasetid = datasetid
        self.class_name=class_name
        self.images = images
        
    def run(self):
        PATH = UPLOAD_FOLDER +'/'+ self.datasetid
        list_classes = os.listdir(os.path.join(PATH))
        if self.class_name in list_classes:
            for i in range(len(self.images)):
                img_name = os.path.join(PATH,self.class_name,self.images[i].filename)
                self.images[i].save(img_name)
            return
        else:
            path= os.path.join(PATH,self.class_name)
            os.mkdir(path)
            for i in range(len(self.images)):
                img_name = os.path.join(PATH,self.class_name,self.images[i].filename)
                self.images[i].save(img_name)
            return


#import data set         
class Import_Data (threading.Thread):
    def __init__(self,filename,datasetid):
        threading.Thread.__init__(self)
        self.filename = filename
        self.dataset = datasetid
    
    def run(self):
        self.savedata(self.filename,self.dataset)
        
    def savedata(self,filename,dataset):
        while global_lock.locked():
            continue

        global_lock.acquire()
        #save file zip
        dataset.save(os.path.join(UPLOAD_FOLDER, filename))
        zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_FOLDER, filename), 'r')
        #extract file zip
        zip_ref.extractall(UPLOAD_FOLDER)
        zip_ref.close()
        os.system('rm -rf '+UPLOAD_FOLDER+'/'+filename)
        global_lock.release()
        
class TrainingThread(threading.Thread):
    def __init__(self,dataset,modelid,batch_size, epochs,lr,types,class_name):
        #threading.Thread.__init__(self)
        super(TrainingThread,self).__init__()
        self.modelname = modelid
        self.dataset = UPLOAD_FOLDER+'/'+dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr=lr
        self.types = types
        self.class_name = class_name
    def run(self):
        if self.types == 'train_dataset':
            
            self.pretrainModel(self.modelname,self.dataset,self.batch_size,self.epochs,self.lr)
        if self.types == 'train_class':
            self.pretrainSingleClass(self.modelname,self.dataset,self.class_name,self.batch_size,self.epochs,self.lr)
    
    def pretrainModel(self,modelname, dataset,batch_size,epochs,lr):
        #K.clear_session()
        graph1 = Graph()
        with graph1.as_default():
            session1 = Session()
            with session1.as_default():
                
                reader = LFWReader(dir_images=dataset)
                gen_train = TripletGenerator(reader)
                gen_test = TripletGenerator(reader)
                embedding_model, triplet_model = GetModel()
                for layer in embedding_model.layers[-3:]:
                    layer.trainable = True
                    
                for layer in embedding_model.layers[: -3]:
                    layer.trainable = False
                triplet_model.compile(loss=None, optimizer=Adam(lr))
                
                history = triplet_model.fit_generator(gen_train, 
                                          validation_data=gen_test,  
                                          epochs=epochs, 
                                          verbose=1,
                                          use_multiprocessing=True,
                                            steps_per_epoch=5,
                                          validation_steps=5)
                embedding_model.save_weights('./trained-models/weights/'+modelname+'.h5')
                self.embeddingmodel(modelname,dataset)
                K.get_session() 
                
        
    def pretrainSingleClass(self,modelname,dataset,class_name,batch_size,epochs,lr):
        #K.clear_session()
        graph2 = Graph()
        with graph2.as_default():
            session2 = Session()
            with session2.as_default():
        
                reader = LFWReader(dir_images=dataset,class_name=class_name)
                gen_train = TripletGeneratorSingleID(reader)
                gen_test = TripletGeneratorSingleID(reader)
                embedding_model, triplet_model = GetModel()
                for layer in embedding_model.layers[-3:]:
                    layer.trainable = True

                for layer in embedding_model.layers[: -3]:
                    layer.trainable = False
                triplet_model.compile(loss=None, optimizer=Adam(lr))

                history = triplet_model.fit_generator(gen_train, 
                                          validation_data=gen_test,  
                                          epochs=epochs, 
                                          verbose=1, 
                                            steps_per_epoch=50,
                                          validation_steps=5)

                embedding_model.save_weights('./trained-models/weights/'+modelname+'.h5')
                self.embeddingmodel(modelname,dataset)
                K.get_session()

        
      
    def embeddingmodel(self,modelname,dataset):
        
        model = VggFace(
            path = './trained-models/weights/'+modelname+'.h5',
                                      is_origin = False)
        metadata = load_metadata(dataset)
        embedded = np.zeros((metadata.shape[0], 2622 ))
        for i in range(metadata.shape[0]):
            img_emb = model.predict(preprocess_image(metadata[i].image_path()))[0,:]
            embedded[i] = img_emb
       
        # save embedding
        np.savetxt('./trained-models/embedding/'+modelname+'_embedded_vector.txt', embedded)
        name = []
        for i in range(len(metadata)):
            name.append(metadata[i].name)
        np.savetxt('./trained-models/embedding/'+modelname+'_name.txt', name,  delimiter=" ", fmt="%s")
        
