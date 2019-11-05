import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os
from libtiff import TIFF

def str2point(point):
    rows = len(point)
    res = np.zeros((rows,2),dtype='int32')
    
    for idx in range(rows):
        res[idx,0], res[idx,1] = point[idx].split(',')
    return res

def isSorted(l,isAcs = True):
    if isAcs:
        return all(l[i] <= l[i+1] for i in range(len(l)-1))
    else:
        return all(l[i] >= l[i+1] for i in range(len(l)-1))
def isAlign(l):
    return all(abs(l[i]-l[i+1])<=30 for i in range(len(l)-1))

def extractPoint(table):
    table_point = table.find('Coords').get('points').split()
    table_point = str2point(table_point)
    
    cell_point = []
    cell_loc = []
    for t in table.iter('cell'):
        tmp = t.find('Coords').get('points').split()
        cell_point.append(str2point(tmp))
        
        sr = int(t.get('start-row'))
        er = int(t.get('end-row'))
        sc = int(t.get('start-col'))
        ec = int(t.get('end-col'))
        cell_loc.append(np.array([sr,er,sc,ec]))
    
    return table_point, cell_point, cell_loc

def checkPoint(point, row, col):
    N = point.shape[0]
    flag = True
    if  N == 2*(row+col):
        vline = point[0:row+1,:]
        flag1 = isSorted(vline[:,1]) and isAlign(vline[:,0])
        hline = point[row:col+row+1,:]
        flag2 = isSorted(hline[:,0]) and isAlign(hline[:,1])
        vline = point[col+row:col+2*row+1,:]
        flag3 = isSorted(vline[:,1], False) and isAlign(vline[:,0])
        hline = point[col+2*row:N,:]
        flag4 = isSorted(hline[:,0], False) and isAlign(hline[:,1])
        if not(flag1 and flag2 and flag3 and flag4):
            flag = False
    else:
        flag = False
    return flag



def preprocess(img_path, xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    rows,cols,ch = img.shape

    img_table = []
    fm_table = []
    
    for table in root.iter('table'):
        table_point, cell_point ,cell_loc = extractPoint(table)
        
        assert len(table_point)>=4, "table point less than 4"
        
        
        idx1 = np.argmin(np.array([x[1]+x[0] for x in table_point])) # top left
        idx2 = np.argmax(np.array([x[1]-x[0] for x in table_point])) # down left
        idx3 = np.argmax(np.array([x[1]+x[0] for x in table_point]))  # down right
        idx4 = np.argmin(np.array([x[1]-x[0] for x in table_point]))  # top right
        tmp = [table_point[idx1], table_point[idx2], table_point[idx3], table_point[idx4]]
        table_point = tmp
        
        table_row = table_point[1][1] - table_point[0][1]
        table_col = table_point[3][0] - table_point[0][0]

        #transformation
        pts1 = np.float32(table_point)
        pts2 = np.float32([[0, 0],[0, table_row],[table_col, table_row],[table_col, 0]])
        M = cv2.getPerspectiveTransform(pts1,pts2)

        #label
        horizon_map = np.zeros((rows,cols) ,dtype="uint8")
        vertical_map = np.zeros((rows,cols) ,dtype="uint8")

        assert len(cell_loc)==len(cell_point)
            
        
        for idx in range(len(cell_point)):
            
            cell_row = cell_loc[idx][1] - cell_loc[idx][0] + 1
            cell_col = cell_loc[idx][3] - cell_loc[idx][2] + 1
            N = cell_point[idx].shape[0]
            
            if checkPoint(cell_point[idx], cell_row, cell_col):          
                assert N==2*(cell_row+cell_col), cell_col               
                # horizontal line feature map
                for i in range(cell_row, cell_row+cell_col):
                    cv2.line(horizon_map, tuple(cell_point[idx][i]), tuple(cell_point[idx][i+1]), 1, 20)           # visuable horizontal line
                for i in range(1, cell_row):
                    cv2.line(horizon_map, tuple(cell_point[idx][i]), tuple(cell_point[idx][N-i-cell_col]), 2, 20)  # unvisuable horizontal line
                for i in range(cell_col+2*cell_row, N-1):
                    cv2.line(horizon_map, tuple(cell_point[idx][i]), tuple(cell_point[idx][i+1]), 1, 20)           # visuable horizontal line
                cv2.line(horizon_map, tuple(cell_point[idx][N-1]), tuple(cell_point[idx][0]), 1, 20)

                # vertical line feature map
                for i in range(cell_row):
                    cv2.line(vertical_map, tuple(cell_point[idx][i]), tuple(cell_point[idx][i+1]), 1, 20)           # visuable vertical line
                for i in range(cell_row+1, cell_col+cell_row):
                    cv2.line(vertical_map, tuple(cell_point[idx][i]), tuple(cell_point[idx][N-i+cell_row]), 2, 20)  # unvisuable vertical line
                for i in range(cell_col+cell_row, cell_col+2*cell_row):
                    cv2.line(vertical_map, tuple(cell_point[idx][i]), tuple(cell_point[idx][i+1]), 1, 20)           # visuable vertical line
            else:
#                 print("%d\t%d\t%d\t%d" %(idx,N,cell_row,cell_col))
                idx1 = np.argmax(np.array([x[1]-x[0] for x in cell_point[idx]]))  # down left
                idx2 = np.argmax(np.array([x[1]+x[0] for x in cell_point[idx]]))  # down right
                idx3 = np.argmin(np.array([x[1]-x[0] for x in cell_point[idx]]))  # top right
                
#                 print(cell_point[idx])
#                 print("%d\t%d\t%d" %(idx1,idx2,idx3))
                
                if idx1>idx2:
                    cell=[]
                    cell.append(cell_point[idx][1:])
                    cell.append(cell_point[idx][0])
                    cell.reverse()
                    idx1 = N-idx1
                    idx2 = N-idx2
                    idx3 = N-idx3
                else:
                    cell = cell_point[idx]
                
                assert N==len(cell), "%d\t%d" %(N, len(cell))
                
                # horizontal line feature map
                for i in range(idx1, idx2):
                    cv2.line(horizon_map, tuple(cell[i]), tuple(cell[i+1]), 1, 20)           # visuable horizontal line
                for i in range(idx3, N-1):
                    cv2.line(horizon_map, tuple(cell[i]), tuple(cell[i+1]), 1, 20)           # visuable horizontal line
                cv2.line(horizon_map, tuple(cell[N-1]), tuple(cell[0]), 1, 20)

                # vertical line feature map
                for i in range(idx1):
                    cv2.line(vertical_map, tuple(cell[i]), tuple(cell[i+1]), 1, 20)           # visuable vertical line
                for i in range(idx2, idx3):
                    cv2.line(vertical_map, tuple(cell[i]), tuple(cell[i+1]), 1, 20)           # visuable vertical line

            
                flag1 = (idx1==cell_row)
                flag2 = ((idx2-idx1)==cell_col)
                flag3 = ((idx3-idx2)==cell_row)
                flag4 = ((N-idx3)==cell_col)
                
                 
                if flag1 and flag3:
                    for i in range(1, idx1):
                        cv2.line(horizon_map, tuple(cell[i]), tuple(cell[idx3-i]), 2, 20)  # unvisuable horizontal line
                else:
                    if flag1:
                        for i in range(1, idx1):
                            cv2.line(horizon_map, tuple(cell[i]), (cell[idx3][0],cell[i][1]), 2, 20)  # unvisuable horizontal line
                    elif flag3:
                        for i in range(idx2+1, idx3):
                            cv2.line(horizon_map, tuple(cell[i]), (cell[idx1][0],cell[i][1]), 2, 20)  # unvisuable horizontal line
                
                if flag2 and flag4:
                    for i in range(idx1+1, idx2):
                        cv2.line(vertical_map, tuple(cell[i]), tuple(cell[N-i+idx1]), 2, 20)  # unvisuable vertical line
                else:
                    if flag2:
                        for i in range(idx1+1, idx2):
                            cv2.line(vertical_map, tuple(cell[i]), (cell[i][0], cell[0][1]), 2, 20)  # unvisuable vertical line
                    elif flag4:
                        for i in range(idx3+1, N-1):
                            cv2.line(vertical_map, tuple(cell[i]), (cell[i][0], cell[idx1][1]), 2, 20)  # unvisuable vertical line
            
#                 cv2.line(vertical_map, tuple(cell_point[idx][0]), tuple(cell_point[idx][idx1]), 250, 20)
#                 cv2.line(horizon_map, tuple(cell_point[idx][idx1]), tuple(cell_point[idx][idx2]), 250, 20)
#                 cv2.line(vertical_map, tuple(cell_point[idx][idx2]), tuple(cell_point[idx][idx3]), 250, 20)
#                 cv2.line(horizon_map, tuple(cell_point[idx][idx3]), tuple(cell_point[idx][0]), 250, 20)

        img_tmp = cv2.warpPerspective(img,M,(table_col,table_row))
        img_table.append(img_tmp)
        
        horizon_map = cv2.warpPerspective(horizon_map,M,(table_col,table_row))
        vertical_map = cv2.warpPerspective(vertical_map,M,(table_col,table_row))
        fm_table.append((horizon_map,vertical_map))
        
        
    return img_table, fm_table