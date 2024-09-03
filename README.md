# Ship-Classification
L&T EduTech Hackathon - Shaastra 2023  
  
Team Name : Bias   
Team Members : <a href="https://github.com/Prathosh-V">Prathosh-V</a>,<a href="https://github.com/Surya-29">Surya-29</a>  
  
PS3:  

Description: Natural disasters and atmospheric anomalies demand remote monitoring and maintenance of naval objects especially big-size ships. For example, under poor weather conditions, prior knowledge about the ship model and type helps the automatic docking system process to be smooth. Thus, this set aims to classify the type of ships from an image data set of ships.   

Requirement Specification:  

Use open source dataset (From the link: https://cutt.ly/PS_3_dataset)  
- Design transfer learning-based CNN architecture to classify the data set  
- Identify an optimal training data size in percentage  
- Judging Metrics: Provide kappa score as Judging metrics  

## Approach & Results:

By levaraging the concept of Transfer learning, a pretrained Xception model, with the top layers of the base model being replaced by a GlobalAveragePooling2D layer and a Dense layer with 5 outputs nodes to classify the input image into five different categories.
  
The Dataset is splitted into 70% train data , 20% validation data and 10% test data for optimal training.  
- After 25 epochs, we attain an __accuracy__ of approx~ __95%-96%__.  
- Also the __Kappa Score__ for the model after training it on 70% train dataset is    
  - <b>Kappa Score : 0.9425842335938575</b>

Final Round Demo:

https://github.com/user-attachments/assets/d5b804cd-22d6-47c3-af31-e3b6f347e4a9



