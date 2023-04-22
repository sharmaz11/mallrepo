subd=temp1.txt
for dfile in ./*.csv; do
	echo $dfile >> $subd 
done

echo "................................."

opn=processed.csv
x=1
for file in $(<$subd); do
	fbn=$(basename "$file")
	echo $fbn
	echo $x$opn
	x=$(( $x + 1 ))

	screen -dm -S "meta"$x python3 metpathway.py $fbn $x$opn
done




rm -f temp1.txt
