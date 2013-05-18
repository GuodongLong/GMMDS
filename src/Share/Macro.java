package Share;

public class Macro {

    /* NRM: 189 documents, (line: 1~189)
     * Labeled_Wiki: 1137 documents, (line:190 ~ 1326)
     * Unlabeled_Wiki: 34557 documents (line:1327 ~ 35883) */
//	public static int SIZEOF_NRM = (int) (189 * Option.trainRatio); //132
//	public static int SIZEOF_LAB_WIKI = 1137;
	public static int SIZEOF_UNL_WIKI = 34557;
	
	/**
	 * The sampled Unlabeled wiki docs. 2000
	 */
	public static int SAMPLED_UNL_WIKI = 2000;
	
	
//	/**
//	 * 189 * 0.7 = 132
//	 */
//	public static int MAX_LINE_LAB_NRM = (int) (189 * Option.trainRatio);
//	
//	/**
//	 * 1269 = 1137 + 132
//	 */
//	public static int MAX_LINE_LAB_WIKI = SIZEOF_NRM + SIZEOF_LAB_WIKI;
//	
//	/**
//	 * 35826 = 34557 + 1269
//	 */
//	public static int MAX_LINE_UNL_WIKI = MAX_LINE_LAB_WIKI + SIZEOF_UNL_WIKI;
	
//	public static void main(String[] args)
//	{
//		System.out.println("SIZEOF_NRM="+SIZEOF_NRM);
//		System.out.println("SIZEOF_LAB_WIKI="+SIZEOF_LAB_WIKI);
//		System.out.println("SIZEOF_UNL_WIKI="+SIZEOF_UNL_WIKI);
//		System.out.println("MAX_LINE_LAB_NRM="+MAX_LINE_LAB_NRM);
//		System.out.println("MAX_LINE_LAB_WIKI="+MAX_LINE_LAB_WIKI);
//		System.out.println("MAX_LINE_UNL_WIKI="+MAX_LINE_UNL_WIKI);
//	}
}
