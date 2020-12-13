![Run Tests](https://github.com/tranthanhtrong/python/workflows/Run%20Tests/badge.svg)
![Discord Webhooks](https://github.com/tranthanhtrong/python/workflows/Discord%20Webhooks/badge.svg)
<br />
<p align="center">
  <a href="https://github.com/tranthanhtrong/python">
    <img src="https://i.ibb.co/3dnVtB5/App-Logo-Inspiraton-156.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Project A</h3>

  <p align="center">
    Feature Selection using Correlation Matrix on Metagenomic Data with Pearson Improving Colorectal Cancer Prediction
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
* [Deadline](#deadline)
* [Contributing](#contributing)
* [License](#license)



<!-- ABOUT THE PROJECT -->
## About The Project

### Built With
This section should list any major library and language:
* [Python](https://www.python.org/)



<!-- GETTING STARTED -->
## Getting Started

### Executation
Run the following command to test all 8 sets of algorithms.
```
python main.py
```
### Current Code Base
 - all the datasets in *.csv
 - 8 file pythons for seperate testing: ```auto_report_a*.py```
 - 1 file python ```a_utils.py``` for contains utilities
 - 8 file pythons for functioning:

```A1./ 70:30 Chéo, RandomForestClassifier to Predict:```
Chạy 3 lần qua 3 phương pháp chọn độ tương quan. Sau đó đem chia 3/7. Dùng RandomForestClassifier to Predict.

```A2./ Như A1, mà tự test trên của nó.```

```A3./ KFold, Kiểm tra Chéo, RandomForestClassifier to Predict.```
Chạy 3 lần qua 3 phương pháp chọn độ tương quan. Sau đó KFOLD và Dùng RandomForestClassifier to Predict.

```A4./ Như A3, mà tự test trên của nó.```

```A5./ KFold, kiểm chéo, SVM to Predict```
Chạy 3 lần qua 3 phương pháp chọn độ tương quan. Sau đó KFOLD và Dùng SVM to Predict.

```A6./ Như A5, mà tự test trên của nó.```

```A7./ 70:30 Chéo, SVM to Predict:```
Chạy 3 lần qua 3 phương pháp chọn độ tương quan. Sau đó đem chia 3/7. Dùng SVM to Predict.

```A8./ Như A7, mà tự test trên của nó.```

<!-- ROADMAP -->
## Deadline
The project should complete before 05/05/2020



<!-- CONTRIBUTING -->
## Contributing

Any contributions you make are **greatly appreciated**.
1. Fork the Project
2. Push to the Branch (`git push origin master`)

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.
