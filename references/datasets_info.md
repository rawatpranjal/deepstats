# Datasets for Deep Learning + Statistical Models Research

## Causal Inference Benchmarks

### dSprites Causal Benchmark
- **Domain**: Image-based causal inference (64×64 sprites)
- **Features**: Disentangled latent parameters (scale, rotation, position)
- **Access**: https://github.com/deepmind/dsprites-dataset
- **Papers**: Xu et al. (2021), Kato et al. (2022)

### ACIC Competition Data (2016-2022)
- **Domain**: Semi-synthetic causal benchmarks
- **Features**: 50-200 covariates with known treatment effects
- **Access**: R packages `aciccomp2016`, `aciccomp2017`
```r
install.packages("aciccomp2016")
library(aciccomp2016)
```

## Survival / Time-to-Event

### TCGA (The Cancer Genome Atlas)
- **Domain**: Cancer genomics
- **Size**: 11,000+ tumors × ~20,000 gene expression features
- **Access**: https://portal.gdc.cancer.gov/
- **Note**: Requires GDC Data Portal account

### MIMIC-IV
- **Domain**: ICU clinical data
- **Size**: 65,000+ ICU stays, time-series vitals + labs
- **Access**: https://physionet.org/content/mimiciv/
- **Note**: Requires credentialing (CITI training)

### MIMIC-CXR
- **Domain**: Chest X-rays + clinical notes
- **Size**: 473,000 images from 45,561 patients
- **Access**: https://physionet.org/content/mimic-cxr/
- **Note**: Requires credentialing

### NASA C-MAPSS (Turbofan Degradation)
- **Domain**: Predictive maintenance / reliability
- **Size**: 100+ engines, 21 sensors × time
- **Direct Download**: https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
- **Alternative**: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

### SUPPORT
- **Domain**: Survival prediction benchmark
- **Size**: 8,873 hospitalized patients, ~40 clinical variables
- **Access**: Available in `pycox` Python package
```python
from pycox.datasets import support
df = support.read_df()
```

## Count / Zero-Inflated Data

### French Motor Third-Party Liability (freMTPL)
- **Domain**: Insurance claims
- **Size**: 678,000 policies × 9 features
- **Access**: R CASdatasets package
```r
install.packages("CASdatasets", repos="http://cas.uqam.ca/pub/")
library(CASdatasets)
data(freMTPL2freq)
data(freMTPL2sev)
```
- **Alternative**: https://www.kaggle.com/datasets/floser/french-motor-claims-datasets-fremtpl2freq

### 10X Genomics PBMC (Single-Cell RNA)
- **Domain**: Single-cell genomics (ZINB data)
- **Size**: 3K, 4K, 8K, 68K cell datasets
- **Access**: https://support.10xgenomics.com/single-cell-gene-expression/datasets
- **Python**: Available in `scanpy` package
```python
import scanpy as sc
adata = sc.datasets.pbmc3k()
```

## Extreme Value / Climate

### GHCN-Daily
- **Domain**: Global weather station data
- **Size**: 100,000+ stations, 1832-present
- **Access**: https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
- **Python**: `noaa-sdk` or direct FTP

### ERA5 Reanalysis
- **Domain**: Global climate reanalysis
- **Size**: Hourly global fields, 1950-present
- **Access**: https://cds.climate.copernicus.eu/
- **Note**: Requires Copernicus Climate Data Store account

## Queueing / Transportation

### NYC TLC Trip Data
- **Domain**: Taxi/rideshare trips
- **Size**: 3+ billion trips since 2009
- **Access**: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- **Format**: Parquet files by month

### NYC 311 Service Requests
- **Domain**: City service call center
- **Size**: Millions of records
- **Access**: https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9

## Software for Analysis

| Model Type | R Package | Python Package |
|------------|-----------|----------------|
| Causal Forest | `grf` | `econml` |
| Survival | `survival`, `grf` | `pycox`, `lifelines` |
| Count/ZINB | `glmmTMB`, `pscl` | `statsmodels` |
| Extreme Value | `extRemes`, `GEVcdn` | `scipy.stats` |
| Single-cell | `Seurat` | `scanpy`, `scvi-tools` |
