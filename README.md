# **Sequence Alignment**

In bioinformatics, a sequence alignment is a way of arranging the sequences of DNA, RNA, or protein to identify regions of similarity that may be a consequence of functional, structural, or evolutionary relationships between the sequences.  [Wikipedia](https://en.wikipedia.org/wiki/Sequence_alignment)

## **Sequence Alignment Evaluation**
Each pair of characters generates a special character that indicates the degree of similarity between them.  
The special characters are `*`, `:`, `.` and ` ` (space).  

* Equal characters will produce a `*`.
* Two characters present in the same conservative group, will produce a `:` sign.
* Two characters present in the same semi-conservative group, will produce a `.`.
* If none of the above is true, the characters will produce a  ` ` (space) sign.

### **Equation**
Since each sign is weighted, the following equation will be performed:  
```W_1 * N_1 + W_2 * N_2 + W_3 * N_3 + W_4 * N_4```  
Where:  
```N_i``` represents the number of time each special character appear.  
```W_i``` represents the wheight that fit to its special character.  


### **Conservative Groups and Semi-Conservative Groups**

| Conservative-Groups | Semi-Conservative Groups |
|---------------------|--------------------------|
| NDEQ NEQK STA       | SAG ATV CSA SGND         |
| MILV QHRK NHQK      | STPA STNK NEQHRK NDEQHK  |
| FYW HY MILF         | SNDEQK HFY FVLIM         |

**Similarity definition**:  
* `Seq2` is places under the Sequence Seq1 with offset `n` from the start of `Seq1`. Where `Seq2` do not allowed to pass behind the end of `Seq1`.
* The letters from `Seq1` that do not have a corresponding letter from `Seq2` are ignored.
* The Alignment Score is calculated according the equation above.


**Examples:**  

1. 
```
seq1 = LQRHKRTHTGEKPYEPSHLQYHERTHTGEKPYECHQCHQAFKKCSLLQRHKRTH
seq2 =                      HERTHTGEKPYECHQCRTAFKKCSLLQRHK
res  =                      ****************: ************
```
Weights: 1.5 2.6 0.3 0.2  
Offset: 21  
Score: 39.2  


2. 
```
seq1 = ELMVRTNMYTONEWVFNVJERVMKLWEMVKL
seq2 =    MSKDVMSDLKWEV
res  =    : .:: :  :* .
```
Weights: 5 4 3 2  
Offset: 3  
Score: -31


## **Mutant Sequence Definition**
For a given Sequence `S` we define a Mutant Sequence ```MS(n)``` which is received by substitution of one or more characters by other character defined by Substitution Rules:
*	The original character is allowed to be substituted by another character if there is no conservative group that contains both characters.  
    For example:
    * `N` is not allowed to be substituted by `H` because both characterss present in conservative group `NHQK`.
    * `N` may be substituted by `W` because there is now conservative group that contains both `N` and `W`.
*   It is not mandatory to substitute all instances of some characters by same substitution character, for example the sequence `PSHLSPSQ` has Mutant Sequence `PFHLSPLQ`.  

## **Project Definition**
For two given sequences ```seq1``` and ```seq2```, find a mutant of ```seq2``` and its offset that produce a maximum / minimum Alignment Score. (will be given as an input as well).  


## **The way to solve the problem**
First, I started writing on paper the solution of the problem in sequentially approach, so that I could understand the problem in a more trivial way, the loops that make up the solution, and all the components of the calculation, so I could understand what could be done in parallel and what could not.  

During the sequential solution, I built the basic algorithms for solving the problem, for example: calculating a relative change in each letter replacement, calculating a result according to the requested formula, etc.  

Also, at this point I developed the administrative functions like reading and processing the information from a file, freeing up allocated memory, etc.  

After I am understandingand developing the sequential solution, I realized that an optimal mutant **can be calculated for each offset individually, and these calculations are independent**, then I can take the best result I found. I wrote myself a very simple example that can illustrate the solution to the problem:  
Assuming we have the following sequences:  
seq1=```𝐴𝐵𝐶𝐷𝐸𝐹𝐺```, seq2=```𝐴𝐵𝐶```  
![image](https://user-images.githubusercontent.com/32777579/130803831-37835853-87ad-48ab-bf86-fcd6ed80ec5a.png)

We can see the job we need to do is equal tomaximum possible offsets, which it equals to ```𝑚𝑎𝑥𝑂𝑓𝑓𝑠𝑒𝑡=𝑙𝑒𝑛(𝑠𝑒𝑞1)−𝑙𝑒𝑛(𝑠𝑒𝑞2)+1```.  
It can be understood that **each offset can be calculated in parallel and there is no dependence** on calculations between each other.

### **Development of the parallel solution**

I used MPI to divide the job between 2 processes (with the possibility that each of them on a different computer), which pass information between them using this tool.  

Both processes received half of ```𝑚𝑎𝑥𝑂𝑓𝑓𝑠𝑒𝑡```.  
Process 1 calculate the optimal mutant in range between ```0``` and  ```𝑚𝑎𝑥𝑂𝑓𝑓𝑠𝑒𝑡/2```(not included), and process two find the optimal mutant between ```𝑚𝑎𝑥𝑂𝑓𝑓𝑠𝑒𝑡/2``` and ```𝑚𝑎𝑥𝑂𝑓𝑓𝑠𝑒𝑡``` (not included).  

After this division both processes sent a calculation range to OpenMP and CUDA.  
Each process divides the work **evenly** between OpenMP and CUDA.

![image](https://user-images.githubusercontent.com/32777579/130805110-f3fb4b92-a8dc-4d32-b1a1-eac1c55e11f2.png)

I chose to implement this solution in **static approach**, little example:

![image](https://user-images.githubusercontent.com/32777579/130805273-5c3b02f4-40fe-42e2-860c-9a0bffbf599d.png)

According to this calculation, the master calculates the offsets between 0 and 5 while OpenMP within the master calculate the offsets between 0 and 2 and CUDA calculates the offsets between 2 and 5.  

The slave does the same as the master, it divides the job between OpenMP and CUDA.  

To implement the algorithm with CUDA, I had to convert a few functions (which initially only worked on the CPU) to run on the GPU.  
Also, in order to decide how many threads and how many blocks I need to implement the parallel work in CUDA, I wrote a function that calculates the optimal number of threads depending on the size of the job to be performed.  
The main purpose of this function is to reduce the number of threads that will not work at all.


### **How to Run**

**Run on one computer:**
1. Within the folder with all resources, open the terminal and type "make".
2. Type "make run file_name={file_name}", where {file_name} is the corresponding input file.

**Run on two computers:**
1. First, you need to edit mf file which includes the the ip of both computers (The first one is the ip of the computer that runs the program, and the second one is the other).
2. Make sure all the resources is in the same folder.
3. On the computer which will run the program, within the folder with all resources, open terminal and type "make".
4. Copy the executable file "prog" from the first computer to the second one. MAKE SURE BOTH EXECUTABLE FILES HAS THE SAME PATH.
5. On the first computer type "make runOn2 file_name={file_name}", where {file_name} is the corresponding input file.
