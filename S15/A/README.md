
# GDrive link for the dataset

https://drive.google.com/drive/folders/1xs2mlCry-5iEAXA-sRQMnmqK-2QjdzDQ

# Dataset Statistics

## Kinds of images 

Image | Type | Channels | Dimension
----- | ---- | -------- | ---------
fg | png | 4 | 112*112
bg | jpg | 3 | 224*224
fg_bg | jpg | 3 | 224*224
fg_bg_masks | jpg | 1 | 224*224
depth | jpg | 1 | 224*224

## Total images of each kind
Image | Count
----- | -----
 fg | 100
 bg | 100
 fg_bg | 400000
 fg_bg_masks | 400000
 depth | 400000

## Total size of the dataset
4.8 GB

## Mean/STD values 
Image | Mean | STD
----- | ---- | ----
fg_bg | [0.5445, 0.5092, 0.4564] | [0.2265, 0.2253, 0.2360]
fg_bg_masks | [0.0571] | [0.2165]
depth images | [0.4385] | [0.2491]

# Dataset Images

## Background
![BG](https://github.com/santhiya-v/EVA/blob/master/S15/A/dataset_sample/bg.png?raw=true)

## Foreground
![FG](https://github.com/santhiya-v/EVA/blob/master/S15/A/dataset_sample/fg.png?raw=true)

## Foreground Mask
![FG Mask](https://github.com/santhiya-v/EVA/blob/master/S15/A/dataset_sample/fg_mask.png?raw=true)

## FG BG
![FG BG](https://github.com/santhiya-v/EVA/blob/master/S15/A/dataset_sample/fg_bg.png?raw=true)

## FG BG Mask
![FG BG Mask](https://github.com/santhiya-v/EVA/blob/master/S15/A/dataset_sample/fg_bg_mask.png?raw=true)

## Depth Images
![Depth](https://github.com/santhiya-v/EVA/blob/master/S15/A/dataset_sample/depth.png?raw=true)

## How dataset was prepared?

* 100 background images were collected
* 100 foreground images were collected. Prefrerred white background and png with transparent background
* Foreground images with white background was made transparent, using GIMP tool. 
  Steps :
    * Open the image in GIMP
    * Select Fuzzy Select Tool and click on white area of the image
    * To add alpha channel - Click on Layer --> Transparency --> Add Alpha Channel 
    * To make white color as transparent - Click on Layer --> Transparency --> Color to alpha --> (Make sure white color is choose in pop-up) --> click OK
    * Now all the white area would have been converted to transparent. Save/Export the image
* Foreground mask was prepared by using opencv.  
    * Alpha (4th) channel of FG alone is created as separate 1 channel mask image. 
* **FG BG Preparation**
    * OpenCV was used
    * FG is overlaid on BG, at (x,y) of BG, using following code :
      ```
      def overlay_transparent(background, overlay, x, y):
            h, w = overlay.shape[0], overlay.shape[1]

            overlay_image = overlay[..., :3]   # First 3 Channel BGR is used as overlay
            mask = overlay[..., 3:] / 255.0    # Fourth channel of overlay is used as mask

            background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

            return background
       ```
    * For one BG, 
        * one FG is taken, and 20 random co-ordinates was generated with in the BG bounds
        * overlay_transparent() method is called for each random co-ordinate, and 20 resulting images are saved
        * FG is then fliped, using opencv, and again the above 2 steps are repeated and 20 images are saved
        * The process is repeated for all FG. At the end, we had 4000 images generated for one BG.
    * Above step was repeated for all 100 BGs and we had 400K images ready
    * Files were written to zip, for easy access. Zip file had 100 folders, where each folder corresponds to 1 BG with 4000 images
    * Code : https://github.com/santhiya-v/EVA/blob/master/S15/A/DatasetPrep_S15.ipynb
    
 * **Dense Depth Images Preparation**
    * When we ran the model for the fg_bg images, prediction was not good, since all our FGs were either subtle or blending with background.
    * Hence sharpened the FGs, before FG BG image dataset was prepared
    * Dense depth load_images code was modified to read images from Zip directly
    * Faced memory error when more images where passed, so prediction was done in chunks, with 250 images at a time.
    * Wrote following code, for processing in chunks :
      ```
      def model_predictions(folder, start, end):
      batch_siz = 250
      iterator_count = int(((end - (start-1)) / batch_siz ))+1
      for j in range(1, iterator_count):
          print("Batch ", j, " started..")
          jstart = (j - 1)*batch_siz + start
          jend = j*batch_siz + (start - 1)
          inputs = load_images(file_name, folder, jstart, jend)
          inputs = scale_up(2, inputs)
          outputs = predict(model, inputs, batch_size=64)
          save_output(outputs, output_dir, folder, jstart, jend)  
      return True

      for i in range(1, 101):
          start_time = time.time()
          start = (i - 1)*4000 + 1
          end = i*4000
          print("Processing BG : ", i)  
          folder = f'{i}'
          model_predictions(folder, start, end)
          end_time = time.time()
          print("--- %s Mins ---" % int((end_time - start_time)/60))
       ```
    * Resulting output was converted to grey scale, since they save space and loading will be easier for the final assignment.
    * For one folder with 4000 images, it took aroung ~6 mins and so we happened to have patience for 10 long hours, to see the final dataset with 400K depth images
    * All the modifications done in dense depth can be found here : https://github.com/santhiya-v/DenseDepth
    
 * **Dataset Statistics** 
      * Wrote a custom Zip dataset loader, using pytorch, to read the images
      * Images where read in batch of 8, mean and std where calculated
      * Code : https://github.com/santhiya-v/EVA/blob/master/S15/A/MeanStd_DataSet.ipynb
      

    
