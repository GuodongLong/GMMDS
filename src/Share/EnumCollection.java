package Share;

import Exception.IllegalEnumValue;

public class EnumCollection {

	public enum LABELS
	{
		Minority("Minority"),
		Majority("Majority");

		String name;
		LABELS(String str)
		{
			this.name = str.substring(0,1) + str.substring(1).toLowerCase();
		}
		
		public String toString()
		{
			return name;
		}
		
		public static int size()
		{
			return 2;
		}
		
		/**
		 * Get the label based on index.
		 * @param index
		 * @return
		 * @throws IllegalEnumValue
		 */
		public static LABELS valueOf(int index) throws IllegalEnumValue
		{
			switch(index)
			{
				case 0: return LABELS.Minority;
				case 1: return LABELS.Majority;
				default: throw new IllegalEnumValue();
			}
		}

		public static int valueOfInt(LABELS label) throws IllegalEnumValue
		{
			if(label.equals(LABELS.Minority))
			{
				return 0;
			}
			if(label.equals(LABELS.Majority))
			{
				return 1;
			}
			throw new IllegalEnumValue();
		}
		
		/**
		 * Compare two labels
		 * @param label1
		 * @param label2
		 * @return
		 */
		public static boolean equal(LABELS label1, LABELS label2)
		{
			if (label1 == label2)
			{
				return true;
			}
			return false;
		}
		
		public static String listString()
		{
			return "Minority:0, Majority:1";
		}
		
		
	}
	
//	public enum LABELS
//	{
//		POSITIVE("1"),
//		NEGATIVE("0");
//
//		String name;
//		LABELS(String str)
//		{
//			if (str.equalsIgnoreCase("0"))
//			{
//				this.name = "NEGATIVE";
//			}
//			else
//			{
//				this.name = "POSITIVE";
//			}
//		}
//		
//		public String toString()
//		{
//			return name;
//		}
//		
//		public static int size()
//		{
//			return 2;
//		}
//		
//		/**
//		 * Get the label based on index.
//		 * @param index
//		 * @return
//		 * @throws IllegalEnumValue
//		 */
//		public static LABELS valueOf(int index) throws IllegalEnumValue
//		{
//			switch(index)
//			{
//				case 0: return LABELS.NEGATIVE;
//				case 1: return LABELS.POSITIVE;
//				default: throw new IllegalEnumValue();
//			}
//		}
//		
//		public static int valueOfInt(LABELS label) throws IllegalEnumValue
//		{
//			if(label.equals(LABELS.NEGATIVE))
//			{
//				return 0;
//			}
//			if(label.equals(LABELS.POSITIVE))
//			{
//				return 1;
//			}
//			throw new IllegalEnumValue();
//		}
//		
//		/**
//		 * Compare two labels
//		 * @param label1
//		 * @param label2
//		 * @return
//		 */
//		public static boolean equal(LABELS label1, LABELS label2)
//		{
//			if (label1 == label2)
//			{
//				return true;
//			}
//			return false;
//		}
//		
//		public static String listString()
//		{
//			return "POSTIVE:1, NEGATIVE:0";
//		}
//	}
}
