# matrix-multiplication
Programming parallel matrix multiplication using sycl

我们使用英特尔oneAPI Developer Cloud 服务用Developer Cloud平台中的CPU与GPU硬件完成并行矩阵乘法。

提交作业使用qsub命令将脚本提交到指定队列上运行 qsub -l nodes=1:gpu:ppn=2 -d . run.sh
