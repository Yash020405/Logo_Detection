# Logo Detection in Videos

This project detects Pepsi and Coca-Cola logos in video files using a YOLOv8 object detection model. It processes the video to identify and track these logos, then outputs an annotated video and a JSON file with details of the detections.


## Setup Instructions
### Clone

- Clone the Repository

    ```bash
    git clone git@github.com:Yash020405/Logo_Detection.git
    ```

### Installation

#### Effectively configure and run this YOLOv8-based logo detection project. 

- Install Dependencies

    Ensure that you have Python and pip installed. 
    It is recommended to set up a virtual environment. 

    ```bash
    python -m venv venv
    ```
    ```bash
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
    ```bash
    pip install -r requirements.txt
    ```

    This will install all the required dependencies to run this model.

### Configure Paths 

To run this project, you will need to set up the path to the video you want to analyze. 

The path should be configured in the script where the video path is set (replace `<path_to_video>` with your actual video path):

```python
video_path = '<path_to_video>'
```
![Screenshot from 2024-07-17 18-03-15](https://github.com/user-attachments/assets/5442dd71-af36-4abb-a5bb-1cb3f5bbed6e)


### Running Model

To run the model, use the following command:

```bash
python main.py
```

## Output 

After running the command, the model will process the input video and generate the following output files in the `Output` folder:

### 1. **Annotated Video**: 
A video file with detected logos annotated.

#### Demo
   This is the link for the demo videos : https://drive.google.com/drive/folders/1Fv7yraqVynzzHx4NmnAy-0uqAMt7R2js?usp=drive_link
### 2. **JSON File**: 
A file containing the timestamp of each logo detection along with their respective height, width, and distance from the center of the frame in pixels.

#### Demo
![Screenshot from 2024-07-17 17-37-16](https://github.com/user-attachments/assets/75997966-7daa-4677-8b1d-53efff548c94)

## Acknowledgements

 - [Ultralytics/YoloV8](https://github.com/ultralytics/ultralytics)
 - [DataSet](https://universe.roboflow.com/thaidd/coca-pepsi-juhuf/dataset/5)


