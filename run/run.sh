echo E0
python3 TransE.py --i 0 --d 100 --e 1500 > out/TransE_0.txt
echo H0
python3 TransH.py --i 0 --d 100 --e 1500 > out/TransH_0.txt
echo M0
python3 DistMult.py --i 0 --d 100 --e 1500 > out/DistMult_0.txt
echo E1
python3 TransE.py --i 1 --d 100 --e 1500 > out/TransE_1:x.txt
echo H1
python3 TransH.py --i 1 --d 100 --e 1500 > out/TransH_1.txt
echo M1
python3 DistMult.py --i 1 --d 100 --e 1500 > out/DistMult_1.txt
echo R0
python3 TransE.py --i 0 --d 100 --e 1000 > out/TransR_00.txt
python3 TransR.py --i 2 --d 100 --e 1000 > out/TransR_01.txt
echo R1
python3 TransE.py --i 1 --d 100 --e 1000 > out/TransR_10.txt
python3 TransR.py --i 2 --d 100 --e 1000 > out/TransR_11.txt
