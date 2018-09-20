
#!/usr/bin/env bash

if [ -z "$1" ]
  then
    echo "Supply your data directory"
else

	plop="$(realpath $1)"
	shift
	./listdatabase/listdatabase.out $plop $@
	sed -i "s|"$plop/"|"''"|g" labels_and_files_of_database.txt 
	sed -i "s|"/"|"' '"|g" labels_and_files_of_database.txt

	if [ -f labels_and_files_of_database_testset.txt ]
	 then  
	   sed -i "s|"$plop/"|"''"|g" labels_and_files_of_database_testset.txt
	   sed -i "s|"/"|"' '"|g" labels_and_files_of_database_testset.txt
	 fi
	
	
    createembedding/createembedding.out $plop ./labels_and_files_of_database.txt $@ -testset labels_and_files_of_database_testset.txt

    #rm labels_and_files_of_database.txt

fi

