from osgeo import gdal
import numpy as np
import cv2


def rgb(_image_pth: str = "", _blue: int = 2, _green: int = 3, _red: int = 4):
    """
    输出BGR三通道真彩色影像。波段编号起始值为1.

    :param _image_pth: 源影像路径。
    :param _blue: 蓝色波段编号。
    :param _green: 绿色波段编号。
    :param _red: 红色波段编号。
    :return: None
    """
    image = gdal.Open(_image_pth)
    trans = image.GetGeoTransform()
    proj = image.GetProjection()

    blue = image.GetRasterBand(_blue)
    blue = blue.ReadAsArray()
    green = image.GetRasterBand(_green)
    green = green.ReadAsArray()
    red = image.GetRasterBand(_red)
    red = red.ReadAsArray()

    # 输出结果到tiff中
    cols = blue.shape[1]
    rows = blue.shape[0]
    driver = gdal.GetDriverByName('Gtiff')
    ds = driver.Create("result/rgb.tif", cols, rows, 3, gdal.GDT_Int16)
    ds.GetRasterBand(1).WriteArray(blue)
    ds.GetRasterBand(2).WriteArray(green)
    ds.GetRasterBand(3).WriteArray(red)
    ds.SetGeoTransform(trans)
    ds.SetProjection(proj)
    ds.FlushCache()
    del ds


def calculate_ndwi(_image_pth: str = "", _green: int = 3, _nir: int = 8, result_path: str = ""):
    """
    计算影像的NDWI。

    :param _image_pth: 源影像路径。
    :param _green: 绿色波段编号。
    :param _nir: 近红外波段编号。
    :return: None
    """
    image = gdal.Open(_image_pth)
    trans = image.GetGeoTransform()
    proj = image.GetProjection()

    green = image.GetRasterBand(_green)
    green = green.ReadAsArray()
    green = green.astype(np.float32, order='C')
    nir = image.GetRasterBand(_nir)
    nir = nir.ReadAsArray()
    nir = nir.astype(np.float32, order='C')

    # 计算ndwi
    _ndwi = (green - nir) / (green + nir)
    # 输出ndwi计算结果到tiff中
    cols = _ndwi.shape[1]
    rows = _ndwi.shape[0]
    driver = gdal.GetDriverByName('Gtiff')
    ds = driver.Create(result_path, cols, rows, 1, gdal.GDT_Float32)
    ds.GetRasterBand(1).WriteArray(_ndwi)
    ds.SetGeoTransform(trans)
    ds.SetProjection(proj)
    ds.FlushCache()
    del ds

    return _ndwi


def reclass(_image_pth: str = ""):
    inp = gdal.Open(_image_pth)
    ndwi_img = inp.GetRasterBand(1).ReadAsArray()
    # 线性拉伸到0-255
    min_ndwi = np.min(ndwi_img)
    max_ndwi = np.max(ndwi_img)
    img = 0 + (ndwi_img - min_ndwi) * (255 - 0) / (max_ndwi - min_ndwi)
    img = img.astype(np.uint8)
    th, b_img = cv2.threshold(img, 0, 1, cv2.THRESH_OTSU)
    with open("result/binary_ndwi_threshold.txt", "wt", encoding="utf-8") as f:
        f.write(f"threshold={th}")
    # cv2.imwrite("result/binary_ndwi.tif", b_img)
    driver = gdal.GetDriverByName('Gtiff')
    ds = driver.Create("result/binary_ndwi.tif", img.shape[1], img.shape[0], 1, gdal.GDT_Int16)
    ds.GetRasterBand(1).WriteArray(b_img)
    ds.SetGeoTransform(inp.GetGeoTransform())
    ds.SetProjection(inp.GetProjection())
    ds.FlushCache()
    del ds


if __name__ == '__main__':
    calculate_ndwi("../data/mini/mini_s2.tif")
    # inp = gdal.Open("result/ndwi_MY.tif")
    # ndwi_img = inp.GetRasterBand(1).ReadAsArray()
    # print()
