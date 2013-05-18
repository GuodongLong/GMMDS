package dataHandling;

public class MDSMatrix {

	public double[][] data;
	public int numLine = 0, numCol = 0;
	
	public MDSMatrix(int l, int c)
	{
		this.numLine = l;
		this.numCol = c;
		this.data = new double[this.numLine][this.numCol];
	}
	
	public void setLine(double[] line, int index)
	{
		this.data[index] = line;
	}
	
	/**
	 * get an array of elements within one line
	 * @param lineId
	 * @return
	 */
	public double[] getLine(int lineId)
	{
		return data[lineId];
	}
	
	/**
	 * get an array of elements within one column
	 * @param colId
	 * @return
	 */
	public double[] getColumn(int colId)
	{
		double[] ret = new double[numLine];
		for(int i = 0; i < numLine; i++)
		{
			ret[i] = data[i][colId];
		}
		return ret;
	}
}
