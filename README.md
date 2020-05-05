# Pixel-Learning [Release branch]
Instructions:

1. Run *python pixel\_train\_1\_query.py config.json*

   config.json must contain: host, port, database, user, passwd.

   Edit the script to change the number of tiles or the starting date for the query.

   This will query the database and return a _toi\_P.csv_ file.

2. Run *python pixel\_train\_2\_learning.py toi\_P.csv output\_dir/*

  This will train the model and save the json outputs in the *output\_dir* folder.

3. Run *python pixel\_train\_3\_merge.py json\_dir/*

  This will merge all the json outputs from step 2 into one big json matrix file, called *weight\_matrix.json*. Move this to the pixel workspace directory on target machine to use it for the predict script.

4. The *pixel_predict.py* script should be called from the pixel service. This will query the database, make the predictions and write them in a csv file.
