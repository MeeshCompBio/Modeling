# Get the datasets from the website
wget --span-hosts http://faculty.marshall.usc.edu/gareth-james/ISL/Advertising.csv -P Data
wget --span-hosts http://faculty.marshall.usc.edu/gareth-james/ISL/Auto.data -P Data
wget --span-hosts http://faculty.marshall.usc.edu/gareth-james/ISL/Auto.csv -P Data
wget --span-hosts http://faculty.marshall.usc.edu/gareth-james/ISL/College.csv -P Data
wget --span-hosts http://faculty.marshall.usc.edu/gareth-james/ISL/Ch10Ex11.csv -P Data
wget --span-hosts http://faculty.marshall.usc.edu/gareth-james/ISL/Credit.csv -P Data
wget --span-hosts http://faculty.marshall.usc.edu/gareth-james/ISL/Income1.csv -P Data
wget --span-hosts http://faculty.marshall.usc.edu/gareth-james/ISL/Income2.csv -P Data
wget --span-hosts http://faculty.marshall.usc.edu/gareth-james/ISL/Heart.csv -P Data

# Get the datasets from the R package
Rscript ISLR_datagen.r