# Parallel Probabilistic Matrix Factorization using C++

## About
Probablistic Matrix Factorization is a class of graphical models commonly used for recommender systems. This project provides a parallel implementation of a Gaussian matrix factorization model utilizing stochastic gradient ascent with no locking to obtain unbiased Maximum A Posteriori (MAP) estimates of the latent user preference and attribute vectors.

## Requirements & Prequisite libraries
* Boost >= 1.7.0
* [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) >=  3.3.9

## Installation
```bash
git clone https://github.com/ageil/parallel-pmf.git
cd parallel-pmf/
cmake .
make
```
To compile & run the unit tests:<br>
```bash
cmake .
make test
```

### Python wrapper
We provide a simple python wrapper library `pmf` to enable interactive analysis, including model recommendations and plottings in jupyter notebooks. To install it:
```bash
cd pypmf
./install.sh
```
Please refer to the [tutorial notebooks](example/pmf_tutorial.md) for details. 

## Running options
```
Parameters for Probabilistic Poisson Matrix Factorization (PPMF):
  -h [ --help ]                         Help
  -i [ --input ] arg (=../data/expr.csv)
                                        Input file name
  -m [ --map ] arg (=../data/genes.csv) Item mapping file name
  -d [ --use_defaults ]                 If enabled, uses './movielens/ratings.c
                                        sv' for the input file and 
                                        './movielens/movies.csv' for the map 
                                        input file
  -o [ --output ] arg                   Output directory
                                        [default: current_path/results/]
                                        
                                        
  --task arg (=train)                   Task to perform
                                        [Options: 'train', 'recommend']
                                        
  -k [ --n_components ] arg (=5)        Number of components (k)
                                        [default: 3]
                                        
  -n [ --n_epochs ] arg (=200)          Num. of learning iterations
                                        [default: 200]
                                        
  -r [ --ratio ] arg (=0.69999999999999996)
                                        Ratio for training/test set splitting
                                        [default: 0.7]
                                        
  --thread arg (=4)                     Number of threads for parallelization
                                        This value must be at least 2
                                        [default: 4]
                                        
  --gamma arg (=0.01)                   Learning rate for gradient descent
                                        [default: 0.01]
                                        
  --lambda_theta arg (=0.050000000000000003)
                                        shape of Gamma prior for spots
                                        
  --lambda_beta arg (=0.050000000000000003)
                                        shape of Gamma prior for genes
                                        
  --eta_theta arg (=10)                 rate of Gamma prior for spots
                                        
  --eta_beta arg (=10)                  rate of Gamma prior for genes
                                        
  -s [ --run_sequential ]               Enable running model fitting 
                                        sequentially
                                        
  --spot                                Recommend items for given spot
                                        
  --item                                Recommend similar items for a given 
                                        item
                                        
  --loss_interval arg (=10)             Number of epochs between each loss 
                                        computation.
                                        [default: 10]

```

## References
- Mnih, A., & Salakhutdinov, R. R. (2007). Probabilistic matrix factorization. *Advances in neural information processing systems*, *20*, 1257-1264
- Niu, F., Recht, B., Ré, C., & Wright, S. J. (2011). Hogwild!: A lock-free approach to parallelizing stochastic gradient descent. arXiv preprint arXiv:1106.5730
- GroupLens Research (2021). MovieLens dataset. https://grouplens.org/datasets/movielens/
