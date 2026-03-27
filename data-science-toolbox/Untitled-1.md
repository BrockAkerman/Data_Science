TCREI

TASK, CONTEXT, REERENCES, EVALUATE, ITERATE
THOUGHTFULLYG CREATE REALLY EXCELLENT INPUTS


```python
import numpy as np
import pandas as pd

from ydata_profiling import ProfileReport
```


```python
pip install ydata-profiling
```

    Collecting ydata-profiling
      Downloading ydata_profiling-4.18.1-py2.py3-none-any.whl.metadata (22 kB)
    Requirement already satisfied: scipy<1.17,>=1.8 in a:\programs\python\lib\site-packages (from ydata-profiling) (1.16.3)
    Requirement already satisfied: pandas!=1.4.0,<3.0,>1.5 in a:\programs\python\lib\site-packages (from ydata-profiling) (2.3.3)
    Collecting matplotlib<=3.10,>=3.5 (from ydata-profiling)
      Downloading matplotlib-3.10.0-cp312-cp312-win_amd64.whl.metadata (11 kB)
    Requirement already satisfied: pydantic<3,>=2 in a:\programs\python\lib\site-packages (from ydata-profiling) (2.12.5)
    Requirement already satisfied: PyYAML<6.1,>=6.0.3 in a:\programs\python\lib\site-packages (from ydata-profiling) (6.0.3)
    Requirement already satisfied: jinja2<3.2,>=3.1.6 in a:\programs\python\lib\site-packages (from ydata-profiling) (3.1.6)
    Collecting visions<0.8.2,>=0.7.5 (from visions[type_image_path]<0.8.2,>=0.7.5->ydata-profiling)
      Downloading visions-0.8.1-py3-none-any.whl.metadata (11 kB)
    Requirement already satisfied: numpy<2.4,>=1.22 in a:\programs\python\lib\site-packages (from ydata-profiling) (2.3.5)
    Collecting minify-html>=0.15.0 (from ydata-profiling)
      Downloading minify_html-0.18.1-cp312-cp312-win_amd64.whl.metadata (18 kB)
    Collecting filetype>=1.0.0 (from ydata-profiling)
      Downloading filetype-1.2.0-py2.py3-none-any.whl.metadata (6.5 kB)
    Requirement already satisfied: phik<0.13,>=0.12.5 in a:\programs\python\lib\site-packages (from ydata-profiling) (0.12.5)
    Requirement already satisfied: requests<3,>=2.32.0 in a:\programs\python\lib\site-packages (from ydata-profiling) (2.32.5)
    Requirement already satisfied: tqdm<5,>=4.66.3 in a:\programs\python\lib\site-packages (from ydata-profiling) (4.67.3)
    Requirement already satisfied: seaborn<0.14,>=0.10.1 in a:\programs\python\lib\site-packages (from ydata-profiling) (0.13.2)
    Collecting multimethod<2,>=1.4 (from ydata-profiling)
      Downloading multimethod-1.12-py3-none-any.whl.metadata (9.6 kB)
    Requirement already satisfied: statsmodels<1,>=0.13.2 in a:\programs\python\lib\site-packages (from ydata-profiling) (0.14.6)
    Collecting typeguard<5,>=4 (from ydata-profiling)
      Downloading typeguard-4.5.1-py3-none-any.whl.metadata (3.8 kB)
    Requirement already satisfied: imagehash==4.3.2 in a:\programs\python\lib\site-packages (from ydata-profiling) (4.3.2)
    Collecting wordcloud>=1.9.4 (from ydata-profiling)
      Downloading wordcloud-1.9.6-cp312-cp312-win_amd64.whl.metadata (3.5 kB)
    Collecting dacite<2,>=1.9 (from ydata-profiling)
      Downloading dacite-1.9.2-py3-none-any.whl.metadata (17 kB)
    Collecting numba<0.63,>=0.60 (from ydata-profiling)
      Downloading numba-0.62.1-cp312-cp312-win_amd64.whl.metadata (2.9 kB)
    Requirement already satisfied: PyWavelets in a:\programs\python\lib\site-packages (from imagehash==4.3.2->ydata-profiling) (1.9.0)
    Requirement already satisfied: pillow in a:\programs\python\lib\site-packages (from imagehash==4.3.2->ydata-profiling) (12.1.1)
    Requirement already satisfied: MarkupSafe>=2.0 in a:\programs\python\lib\site-packages (from jinja2<3.2,>=3.1.6->ydata-profiling) (2.1.5)
    Requirement already satisfied: contourpy>=1.0.1 in a:\programs\python\lib\site-packages (from matplotlib<=3.10,>=3.5->ydata-profiling) (1.3.3)
    Requirement already satisfied: cycler>=0.10 in a:\programs\python\lib\site-packages (from matplotlib<=3.10,>=3.5->ydata-profiling) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in a:\programs\python\lib\site-packages (from matplotlib<=3.10,>=3.5->ydata-profiling) (4.61.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in a:\programs\python\lib\site-packages (from matplotlib<=3.10,>=3.5->ydata-profiling) (1.4.9)
    Requirement already satisfied: packaging>=20.0 in a:\programs\python\lib\site-packages (from matplotlib<=3.10,>=3.5->ydata-profiling) (25.0)
    Requirement already satisfied: pyparsing>=2.3.1 in a:\programs\python\lib\site-packages (from matplotlib<=3.10,>=3.5->ydata-profiling) (3.3.2)
    Requirement already satisfied: python-dateutil>=2.7 in a:\programs\python\lib\site-packages (from matplotlib<=3.10,>=3.5->ydata-profiling) (2.9.0.post0)
    Collecting llvmlite<0.46,>=0.45.0dev0 (from numba<0.63,>=0.60->ydata-profiling)
      Downloading llvmlite-0.45.1-cp312-cp312-win_amd64.whl.metadata (5.0 kB)
    Requirement already satisfied: pytz>=2020.1 in a:\programs\python\lib\site-packages (from pandas!=1.4.0,<3.0,>1.5->ydata-profiling) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in a:\programs\python\lib\site-packages (from pandas!=1.4.0,<3.0,>1.5->ydata-profiling) (2025.2)
    Requirement already satisfied: joblib>=0.14.1 in a:\programs\python\lib\site-packages (from phik<0.13,>=0.12.5->ydata-profiling) (1.1.1)
    Requirement already satisfied: annotated-types>=0.6.0 in a:\programs\python\lib\site-packages (from pydantic<3,>=2->ydata-profiling) (0.7.0)
    Requirement already satisfied: pydantic-core==2.41.5 in a:\programs\python\lib\site-packages (from pydantic<3,>=2->ydata-profiling) (2.41.5)
    Requirement already satisfied: typing-extensions>=4.14.1 in a:\programs\python\lib\site-packages (from pydantic<3,>=2->ydata-profiling) (4.15.0)
    Requirement already satisfied: typing-inspection>=0.4.2 in a:\programs\python\lib\site-packages (from pydantic<3,>=2->ydata-profiling) (0.4.2)
    Requirement already satisfied: charset_normalizer<4,>=2 in a:\programs\python\lib\site-packages (from requests<3,>=2.32.0->ydata-profiling) (3.4.4)
    Requirement already satisfied: idna<4,>=2.5 in a:\programs\python\lib\site-packages (from requests<3,>=2.32.0->ydata-profiling) (3.11)
    Requirement already satisfied: urllib3<3,>=1.21.1 in a:\programs\python\lib\site-packages (from requests<3,>=2.32.0->ydata-profiling) (2.5.0)
    Requirement already satisfied: certifi>=2017.4.17 in a:\programs\python\lib\site-packages (from requests<3,>=2.32.0->ydata-profiling) (2025.10.5)
    Requirement already satisfied: patsy>=0.5.6 in a:\programs\python\lib\site-packages (from statsmodels<1,>=0.13.2->ydata-profiling) (1.0.2)
    Requirement already satisfied: colorama in a:\programs\python\lib\site-packages (from tqdm<5,>=4.66.3->ydata-profiling) (0.4.6)
    Requirement already satisfied: attrs>=19.3.0 in a:\programs\python\lib\site-packages (from visions<0.8.2,>=0.7.5->visions[type_image_path]<0.8.2,>=0.7.5->ydata-profiling) (25.4.0)
    Requirement already satisfied: networkx>=2.4 in a:\programs\python\lib\site-packages (from visions<0.8.2,>=0.7.5->visions[type_image_path]<0.8.2,>=0.7.5->ydata-profiling) (3.6.1)
    Collecting puremagic (from visions<0.8.2,>=0.7.5->visions[type_image_path]<0.8.2,>=0.7.5->ydata-profiling)
      Downloading puremagic-2.0.0-py3-none-any.whl.metadata (7.3 kB)
    Requirement already satisfied: six>=1.5 in a:\programs\python\lib\site-packages (from python-dateutil>=2.7->matplotlib<=3.10,>=3.5->ydata-profiling) (1.17.0)
    Downloading ydata_profiling-4.18.1-py2.py3-none-any.whl (400 kB)
    Downloading dacite-1.9.2-py3-none-any.whl (16 kB)
    Downloading filetype-1.2.0-py2.py3-none-any.whl (19 kB)
    Downloading matplotlib-3.10.0-cp312-cp312-win_amd64.whl (8.0 MB)
       ---------------------------------------- 0.0/8.0 MB ? eta -:--:--
       ---------------------------------------- 8.0/8.0 MB 82.7 MB/s eta 0:00:00
    Downloading minify_html-0.18.1-cp312-cp312-win_amd64.whl (3.1 MB)
       ---------------------------------------- 0.0/3.1 MB ? eta -:--:--
       ---------------------------------------- 3.1/3.1 MB 46.1 MB/s eta 0:00:00
    Downloading multimethod-1.12-py3-none-any.whl (10 kB)
    Downloading numba-0.62.1-cp312-cp312-win_amd64.whl (2.7 MB)
       ---------------------------------------- 0.0/2.7 MB ? eta -:--:--
       ---------------------------------------- 2.7/2.7 MB 40.1 MB/s eta 0:00:00
    Downloading typeguard-4.5.1-py3-none-any.whl (36 kB)
    Downloading visions-0.8.1-py3-none-any.whl (105 kB)
    Downloading wordcloud-1.9.6-cp312-cp312-win_amd64.whl (307 kB)
    Downloading llvmlite-0.45.1-cp312-cp312-win_amd64.whl (38.1 MB)
       ---------------------------------------- 0.0/38.1 MB ? eta -:--:--
       ------------ --------------------------- 11.8/38.1 MB 56.5 MB/s eta 0:00:01
       -------------------- ------------------- 19.4/38.1 MB 47.0 MB/s eta 0:00:01
       -------------------------- ------------- 25.2/38.1 MB 40.8 MB/s eta 0:00:01
       ------------------------------ --------- 29.1/38.1 MB 39.2 MB/s eta 0:00:01
    Note: you may need to restart the kernel to use updated packages.
    

    ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
    
    
    [notice] A new release of pip is available: 24.2 -> 26.0.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
    


```python
import sys
!{sys.executable} -m pip install ydata-profiling
```

    Collecting ydata-profiling
      Using cached ydata_profiling-4.18.1-py2.py3-none-any.whl.metadata (22 kB)
    Requirement already satisfied: scipy<1.17,>=1.8 in a:\programs\python\lib\site-packages (from ydata-profiling) (1.16.3)
    Requirement already satisfied: pandas!=1.4.0,<3.0,>1.5 in a:\programs\python\lib\site-packages (from ydata-profiling) (2.3.3)
    Collecting matplotlib<=3.10,>=3.5 (from ydata-profiling)
      Using cached matplotlib-3.10.0-cp312-cp312-win_amd64.whl.metadata (11 kB)
    Requirement already satisfied: pydantic<3,>=2 in a:\programs\python\lib\site-packages (from ydata-profiling) (2.12.5)
    Requirement already satisfied: PyYAML<6.1,>=6.0.3 in a:\programs\python\lib\site-packages (from ydata-profiling) (6.0.3)
    Requirement already satisfied: jinja2<3.2,>=3.1.6 in a:\programs\python\lib\site-packages (from ydata-profiling) (3.1.6)
    Collecting visions<0.8.2,>=0.7.5 (from visions[type_image_path]<0.8.2,>=0.7.5->ydata-profiling)
      Using cached visions-0.8.1-py3-none-any.whl.metadata (11 kB)
    Requirement already satisfied: numpy<2.4,>=1.22 in a:\programs\python\lib\site-packages (from ydata-profiling) (2.3.5)
    Collecting minify-html>=0.15.0 (from ydata-profiling)
      Using cached minify_html-0.18.1-cp312-cp312-win_amd64.whl.metadata (18 kB)
    Collecting filetype>=1.0.0 (from ydata-profiling)
      Using cached filetype-1.2.0-py2.py3-none-any.whl.metadata (6.5 kB)
    Requirement already satisfied: phik<0.13,>=0.12.5 in a:\programs\python\lib\site-packages (from ydata-profiling) (0.12.5)
    Requirement already satisfied: requests<3,>=2.32.0 in a:\programs\python\lib\site-packages (from ydata-profiling) (2.32.5)
    Requirement already satisfied: tqdm<5,>=4.66.3 in a:\programs\python\lib\site-packages (from ydata-profiling) (4.67.3)
    Requirement already satisfied: seaborn<0.14,>=0.10.1 in a:\programs\python\lib\site-packages (from ydata-profiling) (0.13.2)
    Collecting multimethod<2,>=1.4 (from ydata-profiling)
      Using cached multimethod-1.12-py3-none-any.whl.metadata (9.6 kB)
    Requirement already satisfied: statsmodels<1,>=0.13.2 in a:\programs\python\lib\site-packages (from ydata-profiling) (0.14.6)
    Collecting typeguard<5,>=4 (from ydata-profiling)
      Using cached typeguard-4.5.1-py3-none-any.whl.metadata (3.8 kB)
    Requirement already satisfied: imagehash==4.3.2 in a:\programs\python\lib\site-packages (from ydata-profiling) (4.3.2)
    Collecting wordcloud>=1.9.4 (from ydata-profiling)
      Using cached wordcloud-1.9.6-cp312-cp312-win_amd64.whl.metadata (3.5 kB)
    Collecting dacite<2,>=1.9 (from ydata-profiling)
      Using cached dacite-1.9.2-py3-none-any.whl.metadata (17 kB)
    Collecting numba<0.63,>=0.60 (from ydata-profiling)
      Using cached numba-0.62.1-cp312-cp312-win_amd64.whl.metadata (2.9 kB)
    Requirement already satisfied: PyWavelets in a:\programs\python\lib\site-packages (from imagehash==4.3.2->ydata-profiling) (1.9.0)
    Requirement already satisfied: pillow in a:\programs\python\lib\site-packages (from imagehash==4.3.2->ydata-profiling) (12.1.1)
    Requirement already satisfied: MarkupSafe>=2.0 in a:\programs\python\lib\site-packages (from jinja2<3.2,>=3.1.6->ydata-profiling) (2.1.5)
    Requirement already satisfied: contourpy>=1.0.1 in a:\programs\python\lib\site-packages (from matplotlib<=3.10,>=3.5->ydata-profiling) (1.3.3)
    Requirement already satisfied: cycler>=0.10 in a:\programs\python\lib\site-packages (from matplotlib<=3.10,>=3.5->ydata-profiling) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in a:\programs\python\lib\site-packages (from matplotlib<=3.10,>=3.5->ydata-profiling) (4.61.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in a:\programs\python\lib\site-packages (from matplotlib<=3.10,>=3.5->ydata-profiling) (1.4.9)
    Requirement already satisfied: packaging>=20.0 in a:\programs\python\lib\site-packages (from matplotlib<=3.10,>=3.5->ydata-profiling) (25.0)
    Requirement already satisfied: pyparsing>=2.3.1 in a:\programs\python\lib\site-packages (from matplotlib<=3.10,>=3.5->ydata-profiling) (3.3.2)
    Requirement already satisfied: python-dateutil>=2.7 in a:\programs\python\lib\site-packages (from matplotlib<=3.10,>=3.5->ydata-profiling) (2.9.0.post0)
    Collecting llvmlite<0.46,>=0.45.0dev0 (from numba<0.63,>=0.60->ydata-profiling)
      Using cached llvmlite-0.45.1-cp312-cp312-win_amd64.whl.metadata (5.0 kB)
    Requirement already satisfied: pytz>=2020.1 in a:\programs\python\lib\site-packages (from pandas!=1.4.0,<3.0,>1.5->ydata-profiling) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in a:\programs\python\lib\site-packages (from pandas!=1.4.0,<3.0,>1.5->ydata-profiling) (2025.2)
    Requirement already satisfied: joblib>=0.14.1 in a:\programs\python\lib\site-packages (from phik<0.13,>=0.12.5->ydata-profiling) (1.1.1)
    Requirement already satisfied: annotated-types>=0.6.0 in a:\programs\python\lib\site-packages (from pydantic<3,>=2->ydata-profiling) (0.7.0)
    Requirement already satisfied: pydantic-core==2.41.5 in a:\programs\python\lib\site-packages (from pydantic<3,>=2->ydata-profiling) (2.41.5)
    Requirement already satisfied: typing-extensions>=4.14.1 in a:\programs\python\lib\site-packages (from pydantic<3,>=2->ydata-profiling) (4.15.0)
    Requirement already satisfied: typing-inspection>=0.4.2 in a:\programs\python\lib\site-packages (from pydantic<3,>=2->ydata-profiling) (0.4.2)
    Requirement already satisfied: charset_normalizer<4,>=2 in a:\programs\python\lib\site-packages (from requests<3,>=2.32.0->ydata-profiling) (3.4.4)
    Requirement already satisfied: idna<4,>=2.5 in a:\programs\python\lib\site-packages (from requests<3,>=2.32.0->ydata-profiling) (3.11)
    Requirement already satisfied: urllib3<3,>=1.21.1 in a:\programs\python\lib\site-packages (from requests<3,>=2.32.0->ydata-profiling) (2.5.0)
    Requirement already satisfied: certifi>=2017.4.17 in a:\programs\python\lib\site-packages (from requests<3,>=2.32.0->ydata-profiling) (2025.10.5)
    Requirement already satisfied: patsy>=0.5.6 in a:\programs\python\lib\site-packages (from statsmodels<1,>=0.13.2->ydata-profiling) (1.0.2)
    Requirement already satisfied: colorama in a:\programs\python\lib\site-packages (from tqdm<5,>=4.66.3->ydata-profiling) (0.4.6)
    Requirement already satisfied: attrs>=19.3.0 in a:\programs\python\lib\site-packages (from visions<0.8.2,>=0.7.5->visions[type_image_path]<0.8.2,>=0.7.5->ydata-profiling) (25.4.0)
    Requirement already satisfied: networkx>=2.4 in a:\programs\python\lib\site-packages (from visions<0.8.2,>=0.7.5->visions[type_image_path]<0.8.2,>=0.7.5->ydata-profiling) (3.6.1)
    Collecting puremagic (from visions<0.8.2,>=0.7.5->visions[type_image_path]<0.8.2,>=0.7.5->ydata-profiling)
      Using cached puremagic-2.0.0-py3-none-any.whl.metadata (7.3 kB)
    Requirement already satisfied: six>=1.5 in a:\programs\python\lib\site-packages (from python-dateutil>=2.7->matplotlib<=3.10,>=3.5->ydata-profiling) (1.17.0)
    Using cached ydata_profiling-4.18.1-py2.py3-none-any.whl (400 kB)
    Using cached dacite-1.9.2-py3-none-any.whl (16 kB)
    Using cached filetype-1.2.0-py2.py3-none-any.whl (19 kB)
    Using cached matplotlib-3.10.0-cp312-cp312-win_amd64.whl (8.0 MB)
    Using cached minify_html-0.18.1-cp312-cp312-win_amd64.whl (3.1 MB)
    Using cached multimethod-1.12-py3-none-any.whl (10 kB)
    Using cached numba-0.62.1-cp312-cp312-win_amd64.whl (2.7 MB)
    Using cached typeguard-4.5.1-py3-none-any.whl (36 kB)
    Using cached visions-0.8.1-py3-none-any.whl (105 kB)
    Using cached wordcloud-1.9.6-cp312-cp312-win_amd64.whl (307 kB)
    Downloading llvmlite-0.45.1-cp312-cp312-win_amd64.whl (38.1 MB)
       ---------------------------------------- 0.0/38.1 MB ? eta -:--:--
       ------ --------------------------------- 6.0/38.1 MB 30.7 MB/s eta 0:00:02
       ----------------------------- ---------- 28.3/38.1 MB 71.7 MB/s eta 0:00:01
       ---------------------------------------  38.0/38.1 MB 78.0 MB/s eta 0:00:01
       ---------------------------------------- 38.1/38.1 MB 62.2 MB/s eta 0:00:00
    Downloading puremagic-2.0.0-py3-none-any.whl (65 kB)
    Installing collected packages: minify-html, filetype, typeguard, puremagic, multimethod, llvmlite, dacite, numba, matplotlib, wordcloud, visions, ydata-profiling
      Attempting uninstall: multimethod
        Found existing installation: multimethod 2.0.2
        Uninstalling multimethod-2.0.2:
          Successfully uninstalled multimethod-2.0.2
      Attempting uninstall: matplotlib
        Found existing installation: matplotlib 3.10.8
        Uninstalling matplotlib-3.10.8:
          Successfully uninstalled matplotlib-3.10.8
      Attempting uninstall: visions
        Found existing installation: visions 0.7.4
        Uninstalling visions-0.7.4:
          Successfully uninstalled visions-0.7.4
    Successfully installed dacite-1.9.2 filetype-1.2.0 llvmlite-0.45.1 matplotlib-3.10.0 minify-html-0.18.1 multimethod-1.12 numba-0.62.1 puremagic-2.0.0 typeguard-4.5.1 visions-0.8.1 wordcloud-1.9.6 ydata-profiling-4.18.1
    

    
    [notice] A new release of pip is available: 24.2 -> 26.0.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
    
