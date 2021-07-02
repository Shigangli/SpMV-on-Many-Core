# SpMV-on-Many-Core
A cross-platform Sparse Matrix Vector Multiplication (SpMV) framework for many-core architectures (GPUs and Xeon Phi).


Publication
-----------
The work is pulished in TACO'16, which is an extended work of [yaSpMV](https://dl.acm.org/doi/abs/10.1145/2692916.2555255?casa_token=7c2U3ygKJGsAAAAA:8XKO2rI0mrXN5KsRhMH9jRdACyCBb7jmpthw7ixJEsjKUqbuOM3ipBKwUe2Bf2aUMRP4-_hSx3S4uw). To exploit the performance of Intel Xeon Phi, we extend the BCCOO format by introducing inner-block transpose and propose a new segmented sum/scan to better utilize the 512-bit SIMD instructions. We also add double precision support for SpMV on NVIDIA and AMD GPUs. See the [paper](https://shigangli.github.io/files/TACO-SpMV.pdf) for details. Contact Shigang Li (shigangli.cs@gmail.com) and Shengen Yan (yanshengen@gmail.com) for more questions. 

To cite our work:

```bibtex
@article{zhang2016cross,
  title={A cross-platform SpMV framework on many-core architectures},
  author={Zhang, Yunquan and Li, Shigang and Yan, Shengen and Zhou, Huiyang},
  journal={ACM Transactions on Architecture and Code Optimization (TACO)},
  volume={13},
  number={4},
  pages={1--25},
  year={2016},
  publisher={ACM New York, NY, USA}
}
```

License
-------
See [LICENSE](LICENSE).
