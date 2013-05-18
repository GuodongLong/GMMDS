package matrix;

public class Matrix {

	public int[][] data;
	public int numLine = 0, numCol = 0;
	
	public Matrix(int l, int c)
	{
		this.numLine = l;
		this.numCol = c;
		this.data = new int[this.numLine][this.numCol];
	}
	
	public void setLine(int[] line, int index)
	{
		this.data[index] = line;
	}
}
