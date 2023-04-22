subd=temp1.txt
for dfile in ./*.csv; do
	echo $dfile >> $subd 
done

echo "................................."

opn1=processed1.csv
opn2=processed2.csv
opn3=processed3.csv
x=1
y=1
z=1
for file in $(<$subd); do
	fbn=$(basename "$file")
	echo $fbn
	echo $x$opn
	x=$(( $x + 1 ))
	y=$(( $y + 1 ))
	z=$(( $z + 1 ))

	screen -dm -S "meta"$x python3 metpathway.py $fbn $x$opn1 $y$opn2 $z$opn3
done




rm -f temp1.txt

