Analyst notes for the error analysis, written after reading reports/errors.csv by hand.
report.py embeds everything below the separator into results.md. If the test split
changes, report.py prints a warning because these notes would describe a different run.

test-split-sha256: 017b6ebcd95c46525e9512a593e259c7edcb563a7cfe36588ec292b3f692b6ee

---
**Patterns in the 20 most confident false positives** (legitimate mail scored P ≈ 1.0):

- **Legitimate bulk and commercial mail.** A real Frito-Lay sweepstakes prize notice, Ektron CMS marketing, a CNN Breaking News alert, an AOL Instant Messenger welcome mail and a delivery-failure bounce. They share vocabulary with spam ("Congratulations", "free", "you've won", newsletter boilerplate). The CNN alert also resembles the large spoofed-CNN spam campaign in CEAS_08. Some of these are arguably bulk mail the recipient didn't want, so the "ham" label is debatable.
- **Long forwarded articles on mailing lists.** FoRK-list forwards of BBC, Bollywood and EPA news stories: long, impersonal, no personal context. The model sees no personal signal and leans towards spam.
- **Very short or data-only Enron mails.** A body that is only a URL, or credential lists containing "user id password". The model has almost nothing to go on, and "password" plus a URL are strong phishing cues.

**Patterns in the 20 most confident false negatives** (phishing/spam scored P ≈ 0):

- **Spam that reads like professional mail.** Conference and workshop invitations, chemistry-supplier and industrial-Linux promotions, written in the same register as Enron business mail.
- **Lures with little or no text.** "The file is protected with the password ghj001.", "xJDEHIX Thanks!", and a Nazario phishing body that is only HTML image-map markup. The payload is in an attachment or image the dataset doesn't include, so no text model can catch these. Production would need attachment and HTML analysis.
- **Filter-evasion text.** Word-salad paragraphs and quote collections ("luck is the residue of design…") inserted to dilute spam vocabulary.
- **Label noise.** An academic reply about using the Ling-Spam corpus for discourse analysis is labelled spam, and a SpamAssassin-quarantined message carries the scanner's own report text.

**Takeaways.** Many residual errors sit where the label itself is ambiguous (legitimate bulk mail versus spam), or where the content signal lives outside the text (attachments, images). These are limits of the data and of text-only input, so they don't argue for a bigger text model. Adding sender authentication (SPF/DKIM/DMARC), attachment type and HTML-link analysis is the logical next step. Note that the error rate is highest in SpamAssassin and Ling (§4 table), and that Enron accounts for the largest share of errors (65 of 152).
