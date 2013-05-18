package matrix;

import java.io.FileWriter;
import java.io.IOException;

import Share.log;

public class matrixHelper {

	public static void saveMatrixToFile(Matrix mat, String fileName) throws IOException
	{
		FileWriter fw = new FileWriter(fileName);
		
		for(int i = 0; i < mat.numLine; i++)
		{
			StringBuilder sb = new StringBuilder();
			for(int j = 0; j < mat.numLine; j++)
			{
				sb.append(mat.data[i][j] + " ");
			}
			fw.write(sb.toString());
			fw.write("\n");
		}
		fw.close();
		
		log.print("Write " + mat.numLine + " X " + mat.numCol + " matrix into file " + fileName);
	}
}
