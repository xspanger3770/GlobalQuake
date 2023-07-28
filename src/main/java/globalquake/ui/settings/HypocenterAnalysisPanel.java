package globalquake.ui.settings;
import javax.swing.JSlider;

public class HypocenterAnalysisPanel extends SettingsPanel {
	public HypocenterAnalysisPanel() {
		setLayout(null);
		
		JSlider slider = new JSlider();
		slider.setBounds(56, 12, 200, 16);
		add(slider);
	}

	@Override
	public void save() {
		
	}

	@Override
	public String getTitle() {
		return "Hypocenter Analysis";
	}
}
