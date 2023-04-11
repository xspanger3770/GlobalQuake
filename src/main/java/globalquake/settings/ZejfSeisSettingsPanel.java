package globalquake.settings;

import javax.swing.JLabel;
import javax.swing.JTextField;
import javax.swing.SwingConstants;

import globalquake.core.GlobalQuake;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JCheckBox;

public class ZejfSeisSettingsPanel extends SettingsPanel {

	private static final long serialVersionUID = 1L;
	private JTextField textFieldAddr;
	private JTextField textFieldPort;
	private JCheckBox chckbxReconnect;

	public ZejfSeisSettingsPanel() {
		setLayout(null);

		JLabel lblAddr = new JLabel("IP Address");
		lblAddr.setHorizontalAlignment(SwingConstants.TRAILING);
		lblAddr.setBounds(12, 14, 100, 15);
		add(lblAddr);

		JLabel lblPort = new JLabel("Port");
		lblPort.setHorizontalAlignment(SwingConstants.TRAILING);
		lblPort.setBounds(12, 46, 100, 15);
		add(lblPort);

		textFieldAddr = new JTextField();
		textFieldAddr.setBounds(130, 10, 150, 24);
		textFieldAddr.setText(Settings.zejfSeisIP);
		add(textFieldAddr);
		textFieldAddr.setColumns(10);

		textFieldPort = new JTextField();
		textFieldPort.setBounds(130, 42, 150, 24);
		textFieldPort.setText(Settings.zejfSeisPort + "");
		add(textFieldPort);
		textFieldPort.setColumns(10);

		JButton btnReconnect = new JButton("Reconnect");
		btnReconnect.setBounds(163, 78, 117, 25);
		add(btnReconnect);
		
		btnReconnect.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				if(GlobalQuake.instance != null && GlobalQuake.instance.getZejfClient() != null) {
					GlobalQuake.instance.getZejfClient().reconnect();
				}
			}
		});

		chckbxReconnect = new JCheckBox("Auto reconnect");
		chckbxReconnect.setBounds(12, 79, 143, 23);
		chckbxReconnect.setSelected(Settings.zejfSeisAutoReconnect);
		add(chckbxReconnect);
		// TODO Auto-generated constructor stub
	}

	@Override
	public void save() {
		Settings.zejfSeisIP = textFieldAddr.getText();
		Settings.zejfSeisPort = Integer.valueOf(textFieldPort.getText());
		Settings.zejfSeisAutoReconnect = chckbxReconnect.isSelected();
	}

	@Override
	public String getTitle() {
		return "ZejfSeis";
	}
}
