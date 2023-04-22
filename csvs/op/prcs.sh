subd=temp1.txt
for dfile in ./*.csv; do
	echo $dfile >> $subd
done

cat $subd