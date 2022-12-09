from osgeo import gdal
import numpy as np
from NDWI import calculate_ndwi


def imsave(_array, path: str = "", ref: str = "", bands=1, etype=gdal.GDT_UInt16):
    """
    保存影像。

    :param _array: 影像数据
    :param path: 保存路径
    :param ref: 作为参考系的影像路径或gdal对象
    :param bands: 波段数量
    :param etype: 数据类型
    :return:
    """
    if isinstance(ref, str):
        ref = gdal.Open(ref)
    trans = ref.GetGeoTransform()
    proj = ref.GetProjection()
    driver = gdal.GetDriverByName('Gtiff')
    ds = driver.Create(path, _array.shape[1], _array.shape[0], bands, etype)
    if bands > 1:
        for i in range(1, bands + 1):
            ds.GetRasterBand(i).WriteArray(_array[:, :, i])
    else:
        ds.GetRasterBand(1).WriteArray(_array)
    ds.SetGeoTransform(trans)
    ds.SetProjection(proj)
    ds.FlushCache()
    del ds


def pt_type(_i, _j, image, shape):
    """
    判断像元类型。

    :param _i: 行号
    :param _j: 列号
    :param image: 二值ndwi，np.array
    :param shape: 窗口形状
    :return:
    """
    rows, cols = image.shape[0], image.shape[1]
    dx = int(shape[0] / 2)  # row变化方向
    dy = int(shape[1] / 2)  # col变化方向
    if (_i - dx) < 0:
        if (_j - dy) < 0:
            neighbor = image[0:_i + dx + 1, 0:_j + dy + 1]
        elif (_j + dy + 1) > cols:
            neighbor = image[0:_i + dx + 1, _j - dy:cols]
        else:
            neighbor = image[0:_i + dx + 1, _j - dy:_j + dy + 1]
    elif (_i + dx + 1) > rows:
        if (_j - dy) < 0:
            neighbor = image[_i - dx:rows, 0:_j + dy + 1]
        elif (_j + dy + 1) > cols:
            neighbor = image[_i - dx:rows, _j - dy:cols]
        else:
            neighbor = image[_i - dx:rows, _j - dy:_j + dy + 1]
    else:
        if (_j - dy) < 0:
            neighbor = image[_i - dx:_i + dx + 1, 0:_j + dy + 1]
        elif (_j + dy + 1) > cols:
            neighbor = image[_i - dx:_i + dx + 1, _j - dy:cols]
        else:
            neighbor = image[_i - dx:_i + dx + 1, _j - dy:_j + dy + 1]
    _sum = np.sum(neighbor)
    if _sum == 1:
        # 领域的值全是0
        return 0
    elif _sum == neighbor.shape[0] * neighbor.shape[1]:
        # 领域的值全是1
        return 2
    else:
        # 领域的值不全是0（或1）
        return 1


def growth(image, dem, i, j, n_shape, geo_image, geo_dem, flag=0):
    """
    以递归方式进行区域生长，限制递归层级不大于3.

    :param image: 二值ndwi，np.array
    :param dem: dem，np.array
    :param i: 行号
    :param j: 列号
    :param n_shape: 窗口形状
    :param geo_image: ndwi，gdal对象
    :param geo_dem: dem，gdal对象
    :param flag: 表明当前的递归层级
    :return:
    """
    # 获取image[i,j]点的高程
    h0 = calculate_height(i, j, geo_image, geo_dem, dem)
    # 遍历其邻域
    rows, cols = image.shape[0], image.shape[1]
    dx = int(n_shape[0] / 2)
    dy = int(n_shape[1] / 2)
    xr = [max(i - dx, 0), min(rows, i + dx + 1)]
    yr = [max(j - dy, 0), min(cols, j + dy + 1)]
    weights = np.zeros([3, 3])
    weights = [[0.125, 0.375, 0.125], [0.375, 0, 0.375], [0.125, 0.375, 0.125]]
    for x in range(xr[0], xr[1], 1):
        for y in range(yr[0], yr[1], 1):
            if image[x, y] == 0:
                # 方法一：直接比较非水体点的高程
                # h = calculate_height(x, y, geo_image, geo_dem, dem)
                # if h < h0:
                #     image[x, y] = 1
                #     # 递归image[x,y]
                #     if flag < 10:
                #         growth(image, dem, x, y, n_shape, geo_image, geo_dem, flag + 1)
                # 方法二：统计非水体像元周围属性
                h = calculate_height(x, y, geo_image, geo_dem, dem)
                if h < h0:
                    xrr = [max(x - dx, 0), min(rows, x + dx + 1)]
                    yrr = [max(y - dy, 0), min(cols, y + dy + 1)]
                    neighbor = image[xrr[0]:xrr[1], yrr[0]:yrr[1]]
                    cn = np.sum(neighbor)
                    if cn >= 3:
                        image[x, y] = 1
                        if flag < 100:
                            growth(image, dem, x, y, n_shape, geo_image, geo_dem, flag + 1)


def calculate_height(i, j, image, g_dem, dem):
    """

    :param i: 行号
    :param j: 列号
    :param image: ndwi，np.array
    :param g_dem: dem，gdal对象
    :param dem: dem，np.array
    :return:
    """
    # step 1 计算该像素点(像素正中央)的地理坐标
    [x0, dx, rx, y0, ry, dy] = image.GetGeoTransform()
    Xij = x0 + j * dx + i * rx + dx / 2
    Yij = y0 + i * dy + j * ry + dy / 2
    # step 2 获取该点在dem中的高程
    # 反算地理坐标在DEM中的行列号
    [x0, dx, rx, y0, ry, dy] = g_dem.GetGeoTransform()
    i1 = (Yij * dx - y0 * dx - Xij * ry + x0 * ry + dx * ry / 2 - dx * dy / 2) / (dx * dy - rx * ry)
    j1 = (Xij - x0 - i1 * rx - dx / 2) / dx
    height = dem[int(i1), int(j1)]
    return height


def smooth(ndwi, dem, geo_image, geo_dem, n_shape, result_path):
    """
    通过DEM完善ndwi

    :param ndwi: 浮点ndwi，np.array
    :param dem: 高程，np.array
    :param geo_image: ndwi，gdal对象
    :param geo_dem: dem，gdal对象
    :param n_shape: 生长窗口
    :param result_path: 填充结果输出路径
    :return:
    """
    #
    # 遍历所有值为1的点（即通过ndwi阈值分割认为为水体的点），做如下操作
    rows = ndwi.shape[0]
    cols = ndwi.shape[1]
    for i in range(rows):
        for j in range(cols):
            if ndwi[i, j] == 1:
                p_type = pt_type(i, j, ndwi, n_shape)  # 判断点的类型
                if p_type == 0:
                    # 孤立点，值改为0
                    ndwi[i, j] = 0
                elif p_type == 1:
                    growth(ndwi, dem, i, j, n_shape, geo_image, geo_dem)
    imsave(ndwi, result_path, geo_image)


if __name__ == '__main__':
    # 设置参数
    threshold = -0.05
    dem_path = "../data/mini/qh_dem.tif"
    s2_path = "../data/mini/qh_s2.tif"
    ndwi_path = "result/qh/ndwi.tif"
    b_ndwi_path = "result/qh/b_ndwi.tif"
    filled_path = "result/qh/filled.tif"
    neighbor_shape = [3, 3]
    # 加载DEM
    inp_dem = gdal.Open(dem_path)
    dem = inp_dem.GetRasterBand(1).ReadAsArray()
    # 计算ndwi
    ndwi = calculate_ndwi(_image_pth=s2_path, result_path=ndwi_path)
    # ndwi二值化
    ndwi[ndwi > threshold] = 1
    ndwi[ndwi < threshold] = 0
    ndwi = ndwi.astype(np.uint8)
    imsave(ndwi, b_ndwi_path, s2_path, 1, gdal.GDT_UInt16)  # 保存二值化后的影像

    ref_s2 = gdal.Open(s2_path)

    smooth(ndwi, dem, ref_s2, inp_dem, neighbor_shape, filled_path)
