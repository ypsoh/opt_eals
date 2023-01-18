CFLAGS = -std=c++11 -O3 -Wpedantic -lpthread -march=native -fopenmp
#CFLAGS = -std=c++11 -O0 -Wall -lpthread -march=native -fopenmp

# all: ialspp_main ials_main icd_main popularity_main
all: icd_main icd_main_opt icd_opt_main

ialspp_main: ialspp_main.cc
	g++ -pthread -I eigen-3.3.9 ${CFLAGS} -c ialspp_main.cc -o lib/ialspp_main.o
	g++ -pthread ${CFLAGS} lib/ialspp_main.o -o bin/ialspp_main

ials_main: ials_main.cc
	g++ -pthread -I eigen-3.3.9 ${CFLAGS} -c ials_main.cc -o lib/ials_main.o
	g++ -pthread ${CFLAGS} lib/ials_main.o -o bin/ials_main

icd_main: icd_main.cc
	g++ -pthread -I eigen-3.3.9 ${CFLAGS} -c icd_main.cc -o lib/icd_main.o
	g++ -pthread ${CFLAGS} lib/icd_main.o -o bin/icd_main

icd_main_opt: icd_main_opt.cc
	g++ -pthread -g -I eigen-3.3.9 ${CFLAGS} -c icd_main_opt.cc -o lib/icd_main_opt.o
	g++ -pthread -g ${CFLAGS} lib/icd_main_opt.o -o bin/icd_main_opt

icd_opt_main: icd_main_opt.cc
	g++ -pthread -g -I eigen-3.3.9 ${CFLAGS} -c icd_opt_main.cc -o lib/icd_opt_main.o
	g++ -pthread -g ${CFLAGS} lib/icd_opt_main.o -o bin/icd_opt_main

icd_col_main: icd_col_main.cc
	g++ -pthread -g -I eigen-3.3.9 ${CFLAGS} -c icd_col_main.cc -o lib/icd_col_main.o
	g++ -pthread -g ${CFLAGS} lib/icd_col_main.o -o bin/icd_col_main

popularity_main: popularity_main.cc
	g++ -pthread -I eigen-3.3.9 ${CFLAGS} -c popularity_main.cc -o lib/popularity_main.o
	g++ -pthread ${CFLAGS} lib/popularity_main.o -o bin/popularity_main
