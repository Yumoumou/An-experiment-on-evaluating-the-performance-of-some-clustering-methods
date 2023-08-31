对三种不同的数据（cth3.txt,Spiral.txt,ls3.txt）应用不同的聚类方法(DBSCAN,K-means,AGNES)，并比较不同方法的聚类性能，其中*_cl.txt文件为真实聚类标签。

每组数据都绘制了期望图和应用聚类算法后的结果图；部分算法涉及到超参数选择问题，在本实验中超参数通过手动调试确定；最后使用ARI指数来评估当前聚类算法的性能（ARI取值在-1到1之间，越接近于1则说明聚类结果越准确）。

运行结果可见results.pdf文件。

----------------------------------------------------------------------------------

Applied three clustering methods(DBSCAN,K-means,AGNES)to three different kinds of datasets(cth3.txt,Spiral.txt,ls3.txt), and compared the performances of these clustering methods. Among all the files, *_cl.txt are the true labels of the corresponding datasets.

Drew both the expected results and the predicted results for each datasets. Some of the clustering methods involves hyper-parameter optimization. In this project, the hyper-parameters are decided manually. Finally, the performance of a method is evaluated through ARI, ranging from -1 to 1. The closer the ARI is to 1, the more accurate the clustering result is.

The results are shown in results.pdf.
