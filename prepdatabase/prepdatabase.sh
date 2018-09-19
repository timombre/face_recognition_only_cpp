
#!/usr/bin/env bash

if [ -z "$1" ]
  then
    echo "Supply your data directory"
else

	plop="$(realpath $1)"
	shift
	./listdatabase/listdatabase.out $plop $@
	sed "s|"$plop/"|"''"|g" labels_and_files_of_database.txt > stock.txt
	if [ -f labels_and_files_of_database_testset.txt ]; then sed "s|"$plop/"|"''"|g" labels_and_files_of_database_testset.txt > stock2.txt; fi
	grep "/" stock.txt| sed "s|"/"|"' '"|g" > labels_and_files_of_database.txt
	if [ -f stock2.txt ]; then grep "/" stock2.txt| sed "s|"/"|"' '"|g" > labels_and_files_of_database_testset.txt; fi
	rm stock.txt 
    if [ -f stock2.txt ]; then createembedding/createembedding.out $plop ./labels_and_files_of_database.txt $@; fi

    #rm labels_and_files_of_database.txt

fi

