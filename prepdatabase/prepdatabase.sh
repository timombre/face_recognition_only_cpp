
#!/usr/bin/env bash

if [ -z "$1" ]
  then
    echo "Supply your data directory"
else

	plop="$(realpath $1)"
	./listdatabase/listdatabase.out $plop > labels_and_files_of_database.txt

	sed -i "s|"$plop/"|"''"|g" labels_and_files_of_database.txt 
	grep "/" labels_and_files_of_database.txt| sed "s|"/"|"' '"|g" >plop.txt
	mv plop.txt labels_and_files_of_database.txt

	if [ "$2" ]
	  then
	    ./createembedding/createembedding.out $plop ./labels_and_files_of_database.txt $2
	else
		./createembedding/createembedding.out $plop ./labels_and_files_of_database.txt 

	fi
rm labels_and_files_of_database.txt

fi

