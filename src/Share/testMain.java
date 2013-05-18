package Share;

import java.util.HashMap;
import java.util.Random;

public class testMain {

	public static void main(String[] args)
	{
		HashMap<Integer, Integer> hs = new HashMap<Integer, Integer>(300);
		Random random = new Random(10);
		for(int i = 0; i < 200; i++)
		{
			int index = random.nextInt(200);
			log.print(index + ",");
//			if (hs.get(index) != null)
//			{
//				log.print("repeat index = " + index);
//				continue;
//			}
//			hs.put(index, index);
		}
	}
}
