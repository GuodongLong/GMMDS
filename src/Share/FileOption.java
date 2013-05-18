package Share;


public class FileOption {


	/**
	 * The file store similarity matrix.
	 */
	public static String fileSimMatrix = Comm.dataPath + "all_similarity_matrix.txt";

	/**
	 * The file store labeled wiki data.
	 */
//	public static String NRM_file = Comm.dataPath + "labeled_Nrm\\labeled_nrm_Weka.arff";
	public static String NRM_file = Comm.dataPath + "labeled_Nrm/labeled_nrm_Weka.arff";

	/**
	 * The file store labeled wiki data.
	 */
//	public static String Wiki_file = Comm.dataPath + "labeled_Nrm\\labeled_nwiki_Weka.arff";
	public static String Wiki_file = Comm.dataPath + "labeled_Nrm/labeled_wiki_Weka.arff";
	
	/**
	 * The file store labeled wiki data.
	 */
	public static String NRM_file_2class = Comm.dataPath + "labeled_Nrm\\labeled_nrm_Weka-2class.arff";

	/**
	 * The file store labeled wiki data.
	 */
	public static String NRM_file_nominal = Comm.dataPath + "labeled_Nrm\\labeled_nrm_Weka-nominal.arff";
	
	/**
	 * The file store labeled wiki data.
	 */
	public static String NRM_file_2class_nominal = Comm.dataPath + "labeled_Nrm\\labeled_nrm_Weka-2class-nominal.arff";
	

//	public static String CS_IN_DOMAIN = Comm.dataPath + "20NG\\cs_inDomain.arff";
	public static String CS_IN_DOMAIN = Comm.dataPath + "20NG/cs_inDomain.arff";

//	public static String CS_IN_DOMAIN_Nominal = Comm.dataPath + "20NG\\cs_inDomain-nominal.arff";
	public static String CS_IN_DOMAIN_Nominal = Comm.dataPath + "20NG/cs_inDomain-nominal.arff";


//	public static String CS_OUT_DOMAIN = Comm.dataPath + "20NG\\cs_outDomain.arff";
	public static String CS_OUT_DOMAIN = Comm.dataPath + "20NG/cs_outDomain.arff";

//	public static String CS_OUT_DOMAIN_Nominal = Comm.dataPath + "20NG\\cs_outDomain-nominal.arff";
	public static String CS_OUT_DOMAIN_Nominal = Comm.dataPath + "20NG/cs_outDomain-nominal.arff";
	/**
	 * DataSet: 20 NewsGroup<br>
	 * The target file of Comp vs SCI.<br>
	 * in=target=same-distribution, out=source=diff-distribution;<br>
	 * class label: (0 - Comp, 1 - SCI) or (0 - SCI, 1 - Comp)<br>
	 * 4 Top category: comp, sci, rec, talk<br>
	 * 6 sub-category: cs(comp vs sci),rt(rec vs talk),rs(rec vs sci),st(sci vs talk),cr(comp vs rec), ct(comp vs talk)<br> 
	 */
//	public static String NG_CS_in_File = Comm.dataPath + "public_data\\20NG\\cs_inDomain.arff";
	public static String NG_CS_in_File = Comm.dataPath + "public_data/20NG/cs_inDomain.arff";

	/**
	 * DataSet: 20 NewsGroup<br>
	 * The source file of Comp vs SCI.<br>
	 * in=target=same-distribution, out=source=diff-distribution;<br>
	 * class label: (0 - Comp, 1 - SCI) or (0 - SCI, 1 - Comp)<br>
	 * 4 Top category: comp, sci, rec, talk<br>
	 * 6 sub-category: cs(comp vs sci),rt(rec vs talk),rs(rec vs sci),st(sci vs talk),cr(comp vs rec), ct(comp vs talk)<br> 
	 */
//	public static String NG_CS_out_File = Comm.dataPath + "public_data\\20NG\\cs_outDomain.arff";
	public static String NG_CS_out_File = Comm.dataPath + "public_data/20NG/cs_outDomain.arff";

	/**
	 * DataSet: 20 NewsGroup<br>
	 * The target file of Rec vs Talk.<br>
	 * in=target=same-distribution, out=source=diff-distribution;<br>
	 * class label: (0 - Rec, 1 - Talk) or (0 - Talk, 1 - Rec)<br>
	 * 4 Top category: comp, sci, rec, talk<br>
	 * 6 sub-category: cs(comp vs sci),rt(rec vs talk),rs(rec vs sci),st(sci vs talk),cr(comp vs rec), ct(comp vs talk)<br> 
	 */
//	public static String NG_RT_in_File = Comm.dataPath + "public_data\\20NG\\rt_inDomain.arff";
	public static String NG_RT_in_File = Comm.dataPath + "public_data/20NG/rt_inDomain.arff";

	/**
	 * DataSet: 20 NewsGroup<br>
	 * The target file of Rec vs Talk.<br>
	 * in=target=same-distribution, out=source=diff-distribution;<br>
	 * class label: (0 - Rec, 1 - Talk) or (0 - Talk, 1 - Rec)<br>
	 * 4 Top category: comp, sci, rec, talk<br>
	 * 6 sub-category: cs(comp vs sci),rt(rec vs talk),rs(rec vs sci),st(sci vs talk),cr(comp vs rec), ct(comp vs talk)<br> 
	 */
//	public static String NG_RT_out_File = Comm.dataPath + "public_data\\20NG\\rt_outDomain.arff";
	public static String NG_RT_out_File = Comm.dataPath + "public_data/20NG/rt_outDomain.arff";

	/**
	 * DataSet: Reuters<br>
	 * The source file of Comp vs SCI.<br>
	 * in=target=same-distribution, out=source=diff-distribution;<br>
	 */
//	public static String Reuters_PP_Src_File = Comm.dataPath + "public_data\\Reuters\\PeoplePlaces.src.arff";
	public static String Reuters_PP_Src_File = Comm.dataPath + "public_data/Reuters/PeoplePlaces.src.arff";

	/**
	 * DataSet: Reuters<br>
	 * The source file of Comp vs SCI.<br>
	 * in=target=same-distribution, out=source=diff-distribution;<br>
	 */
//	public static String Reuters_PP_Tar_File = Comm.dataPath + "public_data\\Reuters\\PeoplePlaces.tar.arff";
	public static String Reuters_PP_Tar_File = Comm.dataPath + "public_data/Reuters/PeoplePlaces.tar.arff";
	
	
}
