#h2=1
g2file=final_dat.csv
while [ 1 ]; do
	split -a 4 -d -l 25 --additional-suffix='.csv' $g2file .
done