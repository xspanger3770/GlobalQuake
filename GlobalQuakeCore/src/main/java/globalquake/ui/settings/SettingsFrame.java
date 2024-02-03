package globalquake.ui.settings;

import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.exception.RuntimeApplicationException;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.sounds.Sounds;
import globalquake.ui.GQFrame;
import org.tinylog.Logger;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.util.LinkedList;
import java.util.List;

public class SettingsFrame extends GQFrame {

	private final List<SettingsPanel> panels = new LinkedList<>();
	private final boolean isClient;
	private JTabbedPane tabbedPane;

	public static void main(String[] args) throws Exception{
		TauPTravelTimeCalculator.init();
		Sounds.load();
		GlobalQuake.prepare(new File("./.GlobalQuakeData/"), null);
		EventQueue.invokeLater(() -> {
            try {
                new SettingsFrame(null, false).setVisible(true);
            } catch (Exception e) {
				Logger.error(e);
            }
        });
	}

	// settings panel instance tracker
    private static SettingsFrame openInstance = null;

    public SettingsFrame(Component parent, boolean isClient) {
        this.isClient = isClient;
        // Check if an instance is already open, if so, return before creating a new instance
        if (openInstance != null && openInstance.isVisible()) {
            openInstance.toFront();
            return;
        }
        openInstance = this;
        initialize(parent);
    }

	private void initialize(Component parent) {
		setTitle(!isClient ? "GlobalQuake Settings" : "GlobalQuake Settings (Client)");
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

		JPanel contentPanel = new JPanel(new BorderLayout());
		setContentPane(contentPanel);

		tabbedPane = new JTabbedPane(JTabbedPane.TOP);
		contentPanel.add(tabbedPane, BorderLayout.CENTER);

		JPanel panel_1 = new JPanel();
		contentPanel.add(panel_1, BorderLayout.SOUTH);

		JButton btnCancel = new JButton("Cancel");
		panel_1.add(btnCancel);

		btnCancel.addActionListener(e -> dispose());

		JButton btnSave = new JButton("Save");
		panel_1.add(btnSave);

		btnSave.addActionListener(e -> {
            for (SettingsPanel panel1 : panels) {
                try {
                    panel1.save();
                }
				catch(NumberFormatException exx){
					GlobalQuake.getErrorHandler().handleWarning(new RuntimeApplicationException("Failed to parse a number: %s".formatted(exx.getMessage()), exx));
					return;
				} catch(RuntimeApplicationException exxx){
					GlobalQuake.getErrorHandler().handleWarning(exxx);
					return;
				}
				catch (Exception ex) {
					GlobalQuake.getErrorHandler().handleException(ex);
                    return;
                }
            }
            Settings.save();
			dispose();
        });

		addPanels();

		pack();
		setResizable(false);
		setLocationRelativeTo(parent);
	}

	private void addPanels() {
		panels.add(new GeneralSettingsPanel(this));
		panels.add(new GraphicsSettingsPanel());
		panels.add(new AlertSettingsPanel());
		panels.add(new SoundsSettingsPanel());
		if(!isClient) {
			panels.add(new PerformanceSettingsPanel());
			panels.add(new HypocenterAnalysisSettingsPanel());
		}
		panels.add(new DebugSettingsPanel());

		for (SettingsPanel panel : panels) {
			JScrollPane scrollPane = new JScrollPane(panel);
			scrollPane.setPreferredSize(new Dimension(700, 500));
			scrollPane.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER);
			scrollPane.getVerticalScrollBar().setUnitIncrement(10);
			tabbedPane.addTab(panel.getTitle(), scrollPane);

			javax.swing.SwingUtilities.invokeLater(() -> scrollPane.getVerticalScrollBar().setValue(0));
		}
	}

	public void refreshUI() {
		for (SettingsPanel panel : panels) {
			panel.refreshUI();
		}
	}
}
